import torch
import torch.nn as nn
import torch.nn.functional as F
from .mem_parts.mem_module import Memory
from .mem_parts.ae_module import Encoder, Decoder

__all__ = ['CoMemAE']


class CoMemAE(nn.Module):
    def __init__(self,
                 n_channel=3,
                 t_length=2,
                 motion_memory=10,
                 appearance_memory=10,
                 feature_dim=512,
                 key_dim=512,
                 temp_update=.1,
                 temp_gather=.1):
        super(CoMemAE, self).__init__()

        self.n_channel = n_channel
        # Init Appearence branch
        self.app_encoder = Encoder(t_length=1, n_channels=n_channel)
        self.app_decoder = Decoder(t_length=1, n_channels=n_channel)
        self.app_memory = Memory(appearance_memory, feature_dim, key_dim,
                                 temp_update, temp_gather)

        # Init Motion branch
        self.motion_encoder = Encoder(t_length=t_length, n_channels=n_channel)
        self.motion_decoder = Decoder(t_length=t_length, n_channels=n_channel, networks_type='motion')
        self.motion_memory = Memory(motion_memory, feature_dim, key_dim,
                                    temp_update, temp_gather)

    def forward(self, x, train=True):
        # Seprate input data x into motion block input branch and appeareance
        # input image
        # TODO: Code sepreate input
        app_input = x[:, -self.n_channel:, :, :]
        motion_input = x[:, :-self.n_channel, :, :]
        # Motion branch encoder
        motion_fea, motion_skip1, motion_skip2, motion_skip3 = self.motion_encoder(
                                                                motion_input)
        # Appeareance branch decoder
        app_fea, app_skip1, app_skip2, app_skip3 = self.app_encoder(app_input)

        if train:
            # Motion branch memory
            motion_updated_fea, motion_keys, motion_softmax_score_query, motion_softmax_score_memory, motion_separateness_loss, motion_compactness_loss = self.motion_memory(
                motion_fea, train)
            motion_output = self.motion_decoder(motion_updated_fea,
                                                motion_skip1, motion_skip2,
                                                motion_skip3)

            # Appeareance branch memory
            app_updated_fea, app_keys, app_softmax_score_query, app_softmax_score_memory, app_separateness_loss, app_compactness_loss = self.app_memory(
                app_fea, train)

            app_updated_fea = torch.cat([motion_updated_fea, app_updated_fea], dim=1)
            app_output = self.app_decoder(app_updated_fea, app_skip1,
                                          app_skip2, app_skip3)

            return motion_output, motion_fea, motion_updated_fea, motion_keys,
            motion_softmax_score_query, motion_softmax_score_memory, motion_separateness_loss,
            motion_compactness_loss , app_output, app_fea, app_updated_fea, app_keys,
            app_softmax_score_query, app_softmax_score_memory, app_separateness_loss,
            app_compactness_loss

        else:
            # Motion memory
            motion_updated_fea, motion_keys, motion_softmax_score_query, motion_softmax_score_memory, motion_separateness_loss, motion_compactness_loss = self.motion_memory(
                motion_fea, train)
            motion_output = self.motion_decoder(motion_updated_fea,
                                                motion_skip1, motion_skip2,
                                                motion_skip3)

            # Appeareance branch memory
            app_updated_fea, app_keys, app_softmax_score_query, app_softmax_score_memory, app_separateness_loss, app_compactness_loss = self.app_memory(
                app_fea, train)

            app_updated_fea = torch.cat([motion_updated_fea, app_updated_fea], dim=1)
            app_output = self.app_decoder(app_updated_fea, app_skip1,
                                          app_skip2, app_skip3)

            return motion_output, motion_fea, motion_updated_fea, motion_keys,
            motion_softmax_score_query, motion_softmax_score_memory, motion_separateness_loss,
            motion_compactness_loss , app_output, app_fea, app_updated_fea, app_keys,
            app_softmax_score_query, app_softmax_score_memory, app_separateness_loss,
            app_compactness_loss

def get_model_comemae(cfg):
    if cfg.ARGUMENT.train.normal.use:
        rgb_max = 1.0
    else:
        rgb_max = 255.0
    if cfg.MODEL.flownet == 'flownet2':
        from collections import namedtuple
        from networks.auxiliary.flownet2.models import FlowNet2
        temp = namedtuple('Args', ['fp16', 'rgb_max'])
        args = temp(False, rgb_max)
        flow_model = FlowNet2(args)
        flow_model.load_state_dict(torch.load(cfg.MODEL.flow_model_path)['state_dict'])
    elif cfg.MODEL.flownet == 'liteflownet':
        from networks.auxiliary.liteflownet.models import LiteFlowNet
        flow_model = LiteFlowNet()
        flow_model.load_state_dict({strKey.replace('module', 'net'): weight for strKey, weight in torch.load(cfg.MODEL.flow_model_path).items()})
    else:
        raise Exception('Not support optical flow methods')

    model_dict = OrderedDict()
    model_dict['MemAE'] = CoMemAE(n_channel=cfg.DATASET.channel_num)
    model_dict['FlowNet'] = flow_model

    return model_dict
