import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from .abstract.abstract_hook import EvaluateHook

from datatools.evaluate.utils import psnr_error
from core.utils import flow_batch_estimate, tensorboard_vis_images, save_results, vis_optical_flow
from datatools.evaluate.utils import simple_diff, find_max_patch, amc_score, calc_w

HOOKS = ['COMEMAEEvaluateHook']

class COMEMAEEvaluateHook(EvaluateHook):
    def evaluate(self, current_step):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
        self.trainer.set_requires_grad(self.trainer.CoMemAE, False)
        self.trainer.CoMemAE.eval()
        self.trainer.flownet.eval()
        tb_writer = self.trainer.kwargs['writer_dict']['writer']
        global_steps = self.trainer.kwargs['writer_dict']['global_steps_{}'.format(self.trainer.kwargs['model_type'])]
        frame_num = self.trainer.config.DATASET.test_clip_length
        psnr_records=[]
        score_records=[]

        w_dict = OrderedDict()

        # calc the psnr weight for the test dataset
        for video_name in self.trainer.test_dataset_keys_w:
            dataset = self.trainer.test_dataset_dict_w[video_name]
            len_dataset = dataset.pics_len
            test_iters = len_dataset - frame_num + 1

            test_counter = 0

            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            scores = [0.0 for i in range(len_dataset)]

            for test_data, anno, meta in data_loader:
                # get data
                past_test = data[:, :, 0, :, :].cuda()
                present_test = data[:, :, 1, :, :].cuda()
                future_test = data[:, :, -1, :, :].cuda()

                input_test = torch.cat([past_test, present_test], dim=1)
                # get ground truth optical flow
                gtFlow_vis, gtFlow = flow_batch_estimate(self.trainer.flownet, input_test, self.trainer.normalize.param['val'],
                                                         output_format=self.trainer.config.DATASET.optical_format, optical_size=self.trainer.config.DATASET.optical_size)

                # Run model
                motion_output, _, _, _, motion_softmax_score_query, motion_softmax_score_memory, \
                motion_separateness_loss, motion_compactness_loss , app_output, _, _, _, \
                app_softmax_score_query, app_softmax_score_memory, app_separateness_loss, \
                app_compactness_loss = self.trainer.CoMemAE(input_test, train=False)

                # Cal prediction dif and optical flow
                diff_appe, diff_flow = simple_diff(future_test, app_output, gtFlow, motion_output)
                patch_score_appe, patch_score_flow, _, _ = find_max_patch(diff_appe, diff_flow)
                scores[test_counter+frame_num-1] = [patch_score_appe, patch_score_flow]
                test_counter += 1
                if test_counter >= test_iters:
                    scores[:frame_num-1] = [scores[frame_num-1]]
                    scores = torch.tensor(scores)
                    frame_w =  torch.mean(scores[:,0])
                    flow_w = torch.mean(scores[:,1])
                    w_dict[video_name] = [len_dataset, frame_w, flow_w]
                    print(f'finish calc the scores of training set {video_name} in step:{current_step}')
                    break
        wf, wi = calc_w(w_dict)
        # wf , wi = 1.0, 1.0
        tb_writer.add_text('weight of train set', f'w_f:{wf:.3f}, w_i:{wi:.3f}', global_steps)
        print(f'wf:{wf}, wi:{wi}')
        num_videos = 0
        random_video_sn = torch.randint(0, len(self.trainer.test_dataset_keys), (1,))

        # calc the score for the test dataset
        for sn, video_name in enumerate(self.trainer.test_dataset_keys):
            num_videos += 1
            # need to improve
            dataset = self.trainer.test_dataset_dict[video_name]
            len_dataset = dataset.pics_len
            test_iters = len_dataset - frame_num + 1
            test_counter = 0

            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            vis_range = range(int(len_dataset*0.5), int(len_dataset*0.5 + 5))
            psnrs = np.empty(shape=(len_dataset,),dtype=np.float32)
            scores = np.empty(shape=(len_dataset,),dtype=np.float32)

            for frame_sn, (data, anno, meta) in enumerate(data_loader):
                # get input
                past_test = data[:, :, 0, :, :].cuda()
                present_test = data[:, :, 1, :, :].cuda()
                future_test = data[:, :, -1, :, :].cuda()

                input_test = torch.cat([past_test, present_test], dim=1)
                # get ground truth optical flow
                gtFlow_vis, gtFlow = flow_batch_estimate(self.trainer.flownet, input_test, self.trainer.normalize.param['val'],
                                                         output_format=self.trainer.config.DATASET.optical_format, optical_size=self.trainer.config.DATASET.optical_size)




