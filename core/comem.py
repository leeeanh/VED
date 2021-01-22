#!!!!! ignore the warning messages
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import math
import torch
import time
import numpy as np
from PIL import Image
from collections import OrderedDict
import torchvision.transforms as T
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader

from core.utils import AverageMeter, flow_batch_estimate, vis_optical_flow, tensorboard_vis_images, make_info_message, ParamSet
from datatools.evaluate.utils import psnr_error
from utils.flow_utils import flow2img
from core.engine.default_engine import DefaultTrainer, DefaultInference


class Trainer(DefaultTrainer):
    NAME = ["CoMemAE.TRAIN"]

    def custom_setup(self):
        if self.kwargs['parallel']:
            self.CoMemAE = self.data_parallel(self.model['CoMemAE'])
            self.flownet = self.data_parallel(self.model['FlowNet'])
        else:
            self.CoMemAE = self.model['CoMemAE'].cuda()
            self.flownet = self.model['FlowNet'].cuda()

        # get the optimizer
        self.optim_CoMemAE = self.optimizer['optimizer_comemae']

        # get the loss_fucntion
        self.pred_loss = self.loss_function['rec_loss']
        self.op_loss = self.loss_function['opticalflow_loss']

        # the lr scheduler
        self.lr_comemae = self.lr_scheduler_dict['optimizer_comemae_scheduler']

        # basic meter
        self.loss_meter_CoMemAe = AverageMeter(name='loss_comemae')
        self.loss_meter_pred = AverageMeter(name='loss_pred')
        self.loss_meter_flow = AverageMeter(name='loss_flow')
        self.loss_meter_motion_comp = AverageMeter(name='loss_motion_comp')
        self.loss_meter_app_comp = AverageMeter(name='loss_app_comp')
        self.loss_meter_motion_sep = AverageMeter(name='loss_motion_sep')
        self.loss_meter_app_sep = AverageMeter(name='loss_app_sep')

        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']
        self.test_dataset_keys_w = self.kwargs['test_dataset_keys_w']
        self.test_dataset_dict_w = self.kwargs['test_dataset_dict_w']



    def train(self, current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        self.set_requires_grad(self.CoMemAE, True)
        self.CoMemAE.train()
        self.flownet.eval()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(
            self.kwargs['model_type'])]

        # get data
        data, anno, meta = next(
            self._train_loader_iter)  # the core for dataloader
        self.data_time.update(time.time() - start)

        # base on the D to get each frame
        # in this method, D = 3 and not change
        future = data[:, :, -1, :, :].cuda()  # t+1 frame
        current = data[:, :, 1, :, :].cuda()  # t frame
        past = data[:, :, 0, :, :].cuda()  # t-1 frame

        input_data = data.cuda()

        # True Process =================Start===================
        inputs = torch.cat([past, current], 1)

        # run our model
        motion_output, _, _, _, motion_softmax_score_query, motion_softmax_score_memory, \
        motion_separateness_loss, motion_compactness_loss , app_output, _, _, _, \
        app_softmax_score_query, app_softmax_score_memory, app_separateness_loss, \
        app_compactness_loss = self.CoMemAE(inputs, train=True)

        # get estimated optical flow
        flow_gt_vis, flow_gt = flow_batch_estimate(
            self.flownet,
            inputs,
            self.normalize.param['train'],
            optical_size=self.config.DATASET.optical_size,
            output_format=self.config.DATASET.optical_format)

        # loss
        loss_pred = self.pred_loss(app_output, future)
        loss_motion = self.op_loss(motion_output, flow_gt)
        loss_all = loss_pred + loss_motion + motion_compactness_loss * self.loss_lamada['motion_compactness_loss'] + \
            motion_separateness_loss * self.loss_lamada['motion_separateness_loss'] + \
            app_compactness_loss * self.loss_lamada['app_compactness_loss'] + \
            app_separateness_loss * self.loss_lamada['app_separateness_loss']

        # optimizer
        self.optim_CoMemAE.zero_grad()
        loss_all.backward()
        self.optim_CoMemAE.step()

        # Update meter loss
        self.loss_meter_CoMemAe.update(loss_all.detach())
        self.loss_meter_pred.update(loss_pred.detach())
        self.loss_meter_flow.update(loss_motion.detach())
        self.loss_meter_motion_comp.update(motion_compactness_loss.detach() * self.loss_lamada['motion_compactness_loss'])
        self.loss_meter_app_comp.update(app_compactness_loss.detach() * self.loss_lamada['app_compactness_loss'])
        self.loss_meter_motion_sep.update(motion_separateness_loss.detach() * self.loss_lamada['motion_separateness_loss'])
        self.loss_meter_app_sep.update(app_separateness_loss.detach() * self.loss_lamada['app_separateness_loss'])


        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_comemae.step()
        # ======================End==================

        self.batch_time.update(time.time() - start)

        # Print log
        if (current_step % self.steps.param['log'] == 0):
            msg = make_info_message(current_step, self.steps.param['max'],
                                    self.kwargs['model_type'], self.batch_time,
                                    self.config.TRAIN.batch_size,
                                    self.data_time, [self.loss_meter_CoMemAe,
                                                     self.loss_meter_pred,
                                                     self.loss_meter_flow,
                                                     self.loss_meter_motion_sep,
                                                     self.loss_meter_motion_sep,
                                                     self.loss_meter_app_sep,
                                                     self.loss_meter_app_comp])
            self.logger.info(msg)
        writer.add_scalar('Train_loss_CoMemAE', self.loss_meter_CoMemAe.val,
                          global_steps)

        # Visualization
        if (current_step % self.steps.param['vis'] == 0):
            temp = vis_optical_flow(
                motion_output.detach(),
                output_format=self.config.DATASET.optical_format,
                output_size=(motion_output.shape[-2], motion_output.shape[-1]),
                normalize=self.normalize.param['train'])
            vis_objects = OrderedDict({
                'train_output_app_comemae':
                app_output.detach(),
                'train_output_motion_comemae':
                temp,
                'train_input':
                input_data.detach()
            })
            tensorboard_vis_images(vis_objects, writer, global_steps,
                                   self.normalize.param['train'])
        global_steps += 1

        # reset start
        start = time.time()

        # save model
        self.saved_model = {'MemAE': self.CoMemAE}
        self.saved_optimizer = {'optim_MemAE': self.optim_CoMemAE}
        self.saved_loss = {'loss_MemAE': self.loss_meter_CoMemAe.val}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps

    def mini_eval(self, current_step):
        temp_meter_frame = AverageMeter()
        temp_meter_flow = AverageMeter()
        self.set_requires_grad(self.CoMemAE, False)
        self.CoMemAE.eval()
        for data, _, _ in self.val_dataloader:
            # get the data
            future = data[:, :, -1, :, :].cuda()  # t+1 frame
            current = data[:, :, 1, :, :].cuda()  # t frame
            past = data[:, :, 0, :, :].cuda()  # t-1 frame

            inputs = torch.cat([past, current], dim=1)

            # Run model
            motion_output, _, _, _, motion_softmax_score_query, motion_softmax_score_memory, \
            motion_compactness_loss , app_output, _, _, _, \
            app_softmax_score_query, app_softmax_score_memory, \
            app_compactness_loss = self.CoMemAE(inputs, train=False)

            # Run optical flow estimated
            flow_gt_vis, flow_gt = flow_batch_estimate(
            self.flownet,
            inputs,
            self.normalize.param['train'],
            optical_size=self.config.DATASET.optical_size,
            output_format=self.config.DATASET.optical_format)

            frame_psnr_mini = psnr_error(app_output.detach(), future)
            flow_psnr_mini = psnr_error(motion_output.detach(), flow_gt)
            temp_meter_frame.update(frame_psnr_mini.detach())
            temp_meter_flow.update(flow_psnr_mini.detach())

        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.steps.param["max"]} the frame PSNR is {temp_meter_frame.avg:.3f} the flow PSNR is {temp_meter_flow.avg:.3f}')


class Inference(DefaultInference):
    NAME = ["MEMAE.INFERENCE"]
    def custom_setup(self):
        if self.kwargs['parallel']:
            self.CoMemAE = self.data_parallel(self.model['CoMemAE']).load_state_dict(self.save_model['CoMemAE'])
        else:
            self.MemAE = self.model['CoMemAE'].cuda()
            self.MemAE.load_state_dict(self.save_model['CoMemAE'])

        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']


    def inference(self):
        for h in self._hooks:
            h.inference()
