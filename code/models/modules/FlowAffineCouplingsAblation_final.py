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

import torch
from torch import nn as nn

from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros
from utils.util import opt_get



class CondAffineSeparatedAndCond_1dflow(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 3
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        # hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 #if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'],  0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2
        # print('channels_for_nn',self.channels_for_nn)
        self.fAffine1 = self.F(in_channels=self.channels_for_nn+32,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)
        self.fAffine1_ = self.F(in_channels=self.channels_for_nn+32,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)
    def forward(self, input: torch.Tensor, logdet=None, reverse=False,cond_fea=None):

        if not reverse:
            z = input

            # print(z.max(),z.min())
            # Self Conditional
            # self.condfeas = self.inputextract(cond_fea)
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1,self.fAffine1,cond_fea)

            z2 = z2 + shift
            z2 = z2 * scale
            logdet = logdet + self.get_logdet(scale)

            scale_, shift_ = self.feature_extract_aff(z2,self.fAffine1_,cond_fea)     
            z1 = z1 + shift_
            z1 = z1 * scale_

            logdet = logdet + self.get_logdet(scale_)
            # print('process',z2.max())
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input
            # print(z)
            z1, z2 = self.split(z)
            # print('original',z.shape)
            # self.condfeas = self.inputextract(cond_fea)
            scale, shift = self.feature_extract_aff(z2,self.fAffine1_,cond_fea)

            logdet = logdet - self.get_logdet(scale)
            # self.asserts(scale, shift, z1, z2)
            z1 = z1 / scale
            z1 = z1 - shift

            scale_, shift_ = self.feature_extract_aff(z1,self.fAffine1,cond_fea)

            # self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale_
            z2 = z2 - shift_

            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale_)

            output = z

        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])



    def feature_extract_aff(self, z, f,condfea):


        z1 = torch.cat((z,condfea),1)

        h = f(z1)
        shift, scale = thops.split_feature(h, "cross")
        # scale = (torch.sigmoid(scale+2.0) + 0.0001)
        scale = (1.5*torch.sigmoid(scale) + 0.5)
        scale = scale 
        shift = shift 
        return scale, shift


    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=[1, 1])]
        layers.append(nn.ReLU(inplace=False))
        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=[1, 1]))
        return nn.Sequential(*layers)

class CondAffineSeparated_1dflow(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 3
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        # hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 #if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'],  0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2
        # print('channels_for_nn',self.channels_for_nn)
        self.fAffine1nocond = self.F(in_channels=self.channels_for_nn,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)
        self.fAffine1nocond_ = self.F(in_channels=self.channels_for_nn,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)
    def forward(self, input: torch.Tensor, logdet=None, reverse=False,cond_fea=None):

        if not reverse:
            z = input

            # print(z.max(),z.min())
            # Self Conditional
            # self.condfeas = self.inputextract(cond_fea)
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1,self.fAffine1nocond)

            z2 = z2 + shift
            z2 = z2 * scale
            logdet = logdet + self.get_logdet(scale)

            scale_, shift_ = self.feature_extract_aff(z2,self.fAffine1nocond_)     
            z1 = z1 + shift_
            z1 = z1 * scale_

            logdet = logdet + self.get_logdet(scale_)
            # print('process',z2.max())
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input
            # print(z)
            z1, z2 = self.split(z)
            # print('original',z.shape)
            # self.condfeas = self.inputextract(cond_fea)
            scale, shift = self.feature_extract_aff(z2,self.fAffine1nocond_)

            logdet = logdet - self.get_logdet(scale)
            # self.asserts(scale, shift, z1, z2)
            z1 = z1 / scale
            z1 = z1 - shift

            scale_, shift_ = self.feature_extract_aff(z1,self.fAffine1nocond)

            # self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale_
            z2 = z2 - shift_

            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale_)

            output = z

        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])



    def feature_extract_aff(self, z, f):


        z1 = z#torch.cat((z,condfea),1)

        h = f(z1)
        shift, scale = thops.split_feature(h, "cross")
        # scale = (torch.sigmoid(scale+2.0) + 0.0001)
        scale = (1.5*torch.sigmoid(scale) + 0.5)
        scale = scale 
        shift = shift 
        return scale, shift


    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=[1, 1])]
        layers.append(nn.ReLU(inplace=False))
        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=[1, 1]))
        return nn.Sequential(*layers)