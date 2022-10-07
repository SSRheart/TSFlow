# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import numpy as np
import torch
from torch import nn as nn

import models.modules.Split
from models.modules import flow, thops
from models.modules.Split import Split2d
from models.modules.glow_arch import f_conv2d_bias
# from models.modules.FlowStep1Dbi import FlowStep
from models.modules.FlowStep1D import FlowStep

from utils.util import opt_get

class Flow1dNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L=None,
                 actnorm_scale=1.0,
                 flow_permutation=None,
                 flow_coupling="affine",
                 LU_decomposed=False, opt=None):

        super().__init__()
        # self.scale = Scale(dim=(1,3,384,384))
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.L = opt_get(opt, ['network_G', 'flow', 'L'])
        self.K = opt_get(opt, ['network_G', 'flow', 'K'])
        if isinstance(self.K, int):
            self.K = [K for K in [K, ] * (self.L + 1)]

        self.opt = opt
        H, W, self.C = image_shape
        # self.check_image_shape()

        self.ToName = {0:'fea_1',
                   1:'fea_2',
                   2:'fea_3',
                   3:'fea_4',
                   4:'fea_5',
                   5:'fea_6',
                   6:'fea_7',
                   7:'fea_8'
                   }

        affineInCh = self.get_affineInCh(opt_get)
        flow_permutation = self.get_flow_permutation(flow_permutation, opt)

        normOpt = opt_get(opt, ['network_G', 'flow', 'norm'])

        conditional_channels = {}
        n_rrdb = self.get_n_rrdb_channels(opt, opt_get)
        n_bypass_channels = opt_get(opt, ['network_G', 'flow', 'levelConditional', 'n_channels'])
        conditional_channels[0] = n_rrdb
        for level in range(1, self.L + 1):

            n_bypass = 0 if n_bypass_channels is None else (self.L - level) * n_bypass_channels
            conditional_channels[level] = n_rrdb + n_bypass
        # Upsampler
        for level in range(1, self.L + 1):

            self.arch_FlowStep(H, K, LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling,
                               flow_permutation,
                               hidden_channels, normOpt, opt, opt_get,
                               n_conditinal_channels=conditional_channels[level])
        

        if opt_get(opt, ['network_G', 'flow', 'split', 'enable']):
            self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64 // 2 // 2)
        else:
            self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64)

        self.H = H
        self.W = W
        self.scaleH = 160 / H
        self.scaleW = 160 / W

    def get_n_rrdb_channels(self, opt, opt_get):
        blocks = opt_get(opt, ['network_G', 'flow', 'stackRRDB', 'blocks'])
        n_rrdb = 64 if blocks is None else (len(blocks) + 1) * 64
        return n_rrdb

    def arch_FlowStep(self, H, K, LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling, flow_permutation,
                      hidden_channels, normOpt, opt, opt_get, n_conditinal_channels=None):
        condAff = self.get_condAffSetting(opt, opt_get)
        if condAff is not None:
            condAff['in_channels_rrdb'] = n_conditinal_channels

        for k in range(0,K):
            position_name = get_position_name(H, self.opt['scale'])
            if normOpt: normOpt['position'] = position_name
            self.layers.append(
                FlowStep(in_channels=self.C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling='globalfeature',
                        acOpt=condAff,
                        position=position_name,
                        LU_decomposed=LU_decomposed, opt=opt, idx=k, normOpt=normOpt))
            self.output_shapes.append(
                [-1, self.C, H, W])


    def get_condAffSetting(self, opt, opt_get):
        condAff = opt_get(opt, ['network_G', 'flow', 'condAff']) or None
        condAff = opt_get(opt, ['network_G', 'flow', 'condFtAffine']) or condAff
        return condAff



    def arch_additionalFlowAffine(self, H, LU_decomposed, W, actnorm_scale, hidden_channels, opt):
        if 'additionalFlowNoAffine' in opt['network_G']['flow']:
            n_additionalFlowNoAffine = int(opt['network_G']['flow']['additionalFlowNoAffine'])
            for _ in range(n_additionalFlowNoAffine):
                self.layers.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation='invconv',
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed, opt=opt))
                self.output_shapes.append(
                    [-1, self.C, H, W])

    def arch_squeeze(self, H, W):
        self.C, H, W = self.C * 4, H // 2, W // 2
        self.layers.append(flow.SqueezeLayer(factor=2))
        self.output_shapes.append([-1, self.C, H, W])
        return H, W
    def arch_unsqueeze(self, H, W):
        self.C, H, W = self.C // 4, H * 2, W * 2
        self.layers.append(flow.UnsqueezeLayer(factor=2))
        self.output_shapes.append([-1, self.C, H, W])
        return H, W


    def get_flow_permutation(self, flow_permutation, opt):
        flow_permutation = opt['network_G']['flow'].get('flow_permutation', 'invconv')
        return flow_permutation

    def get_affineInCh(self, opt_get):
        affineInCh = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        affineInCh = (len(affineInCh) + 1) * 64
        return affineInCh

    def check_image_shape(self):
        assert self.C == 1 or self.C == 3, ("image_shape should be HWC, like (64, 64, 3)"
                                            "self.C == 1 or self.C == 3")

    def forward(self, gt=None, z=None, epses=None, logdet=0., reverse=False, eps_std=None,
                y_onehot=None,cond_fea=None):
        # globalfeature = gt
        # print('Flow1dNetFlow1dNetFlow1dNetFlow1dNetFlow1dNetFlow1dNet')
        if reverse:

            sr, logdet = self.decode(gt, z, logdet=logdet,cond_fea=cond_fea)

            return sr, logdet
        else:

            z, logdet = self.encode(gt, logdet=logdet,cond_fea=cond_fea)

            return z, logdet

    def encode(self, globalfeature, logdet=0.0,cond_fea=None):
        # fl_fea = gt
        epses = []
        fl_fea = globalfeature#-rrdbResults[self.ToName[7]] #rrdbResults[self.ToName[8]]
        reverse = False
        level_conditionals = {}
        # bypasses = {}
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        L = opt_get(self.opt, ['network_G', 'flow', 'L'])
        level = 7 
        epses.append(fl_fea)
        for layer, shape in zip(self.layers, self.output_shapes):
            size = shape[2]
            # print(layer)
            # print('fl_fea',fl_fea.max(),fl_fea.min())
            fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse,cond_fea=cond_fea)
        epses.append(fl_fea)
        return epses[-1], logdet#+loss_k
    def get_logdet(self, scale):

        return thops.sum(torch.log(scale), dim=[1, 2, 3])*3
    def forward_preFlow(self, fl_fea, logdet, reverse):
        if hasattr(self, 'preFlow'):
            for l in self.preFlow:
                fl_fea, logdet = l(fl_fea, logdet, reverse=reverse)
        return fl_fea, logdet

    def forward_split2d(self, epses, fl_fea, layer, logdet, reverse, rrdbResults, y_onehot=None):
        ft = None if layer.position is None else rrdbResults[layer.position]
        fl_fea, logdet, eps = layer(fl_fea, logdet, reverse=reverse, eps=epses, ft=ft, y_onehot=y_onehot)
        if isinstance(epses, list):
            epses.append(eps)
        return fl_fea, logdet

    def decode(self, globalfeature, z, logdet=0.0,cond_fea=None):
        # z = epses.pop() if isinstance(epses, list) else z
        # z = torch.normal(mean=0, std=0.8, size=(1, 64, 1, 1)).cuda() 
        fl_fea = z
        # print('cond_fea',cond_fea.shape)
        bypasses = {}
        epses = []
        for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):

            fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True,cond_fea=cond_fea)
            # elif isinstance(layer,Scale):
                # fl_fea = layer(fl_fea, inverse=True)
        sr = fl_fea#+rrdbResults[self.ToName[7]]
        return sr, globalfeature

    def forward_split2d_reverse(self, eps_std, epses, fl_fea, layer, rrdbResults, logdet, y_onehot=None):
        ft = None if layer.position is None else rrdbResults[layer.position]
        fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True,
                               eps=epses.pop() if isinstance(epses, list) else None,
                               eps_std=eps_std, ft=ft, y_onehot=y_onehot)
        return fl_fea, logdet


def get_position_name(H, scale):
    downscale_factor = 160 // H
    position_name = 'fea_up{}'.format(scale / downscale_factor)
    return position_name
class Scale(nn.Module):
    def __init__(self, dim=[1,32,1,1], kernel=None):
        super(Scale, self).__init__()
        if kernel is None:
            self.kernel = nn.Parameter(torch.ones(dim).repeat(8,1,1,1))
            nn.init.xavier_normal_(self.kernel)
        else:
            self.kernel = nn.Parameter(kernel)

    def forward(self, x, inverse=False):
        # self.kernel = torch.cat((self.kernel,self.kernel,self.kernel,self.kernel),0)
        # print('scale')
        if inverse:
            # print(x.shape)
            return torch.exp(-self.kernel[0:1,:,:,:]) * x
        else:
            return torch.exp(self.kernel)* x,self.kernel
