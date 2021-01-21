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

                # Run model
                motion_output, _, _, _, motion_softmax_score_query, motion_softmax_score_memory, \
                motion_separateness_loss, motion_compactness_loss , app_output, _, _, _, \
                app_softmax_score_query, app_softmax_score_memory, app_separateness_loss, \
                app_compactness_loss = self.trainer.CoMemAE(input_test, train=False)

                test_psnr = psnr_error(app_output, future_test)
                score, _, _ = amc_score(future_test, app_output, gtFlow, motion_output, wf, wi)
                test_psnr = test_psnr.tolist()
                score = score.tolist()
                psnrs[test_counter+frame_num-1] = test_psnr
                scores[test_counter+frame_num-1] = score
                test_counter += 1

                if sn == random_video_sn and (frame_sn in vis_range):
                    temp = vis_optical_flow(motion_output.detach(), output_format=self.trainer.config.DATASET.optical_format, output_size=(motion_output.shape[-2], motion_output.shape[-1]),
                                            normalize=self.trainer.normalize.param['val'])
                    vis_objects = OrderedDict({
                        'comem_eval_frame': future_test.detach(),
                        'comem_eval_frame_hat': app_output.detach(),
                        'comem_eval_flow': gtFlow_vis.detach(),
                        'comem_eval_flow_hat': temp
                    })
                    tensorboard_vis_images(vis_objects, tb_writer, global_steps, normalize=self.trainer.normalize.param['val'])

                if test_counter >= test_iters:
                    psnrs[:frame_num-1]=psnrs[frame_num-1]
                    # import ipdb; ipdb.set_trace()
                    scores[:frame_num-1]=(scores[frame_num-1],) # fix the bug: TypeError: can only assign an iterable
                    smax = max(scores)
                    normal_scores = np.array([np.divide(s, smax) for s in scores])
                    normal_scores = np.clip(normal_scores, 0, None)
                    psnr_records.append(psnrs)
                    score_records.append(normal_scores)
                    print(f'finish test video set {video_name}')
                    break

        self.trainer.pkl_path = save_results(self.trainer.config, self.trainer.logger, verbose=self.trainer.verbose, config_name=self.trainer.config_name, current_step=current_step, time_stamp=self.trainer.kwargs["time_stamp"],score=score_records, psnr=psnr_records)
        results = self.trainer.evaluate_function(self.trainer.pkl_path, self.trainer.logger, self.trainer.config, self.trainer.config.DATASET.score_type)
        self.trainer.logger.info(results)
        tb_writer.add_text('CoMemAE: AUC of ROC curve', f'auc is {results.auc}',global_steps)
        return results.auc


def get_comem_hooks(name):
    if name in HOOKS:
        t = eval(name)()
    else:
        raise Exception('The hook is not in amc_hooks')
    return t
