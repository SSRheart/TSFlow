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

import models.modules
import models.modules.Permutations
from models.modules import flow, thops,FlowAffineCouplingsAblation_final
from utils.util import opt_get


def getConditional(rrdbResults, position):
    img_ft = rrdbResults if isinstance(rrdbResults, torch.Tensor) else rrdbResults[position]
    return img_ft

class FlowStep(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "squeeze_invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_alternating_2_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlign": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1SubblocksShuf": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder4": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive",
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None):
        # check configures
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector

        self.norm_type = normOpt['type'] if normOpt else 'ActNorm2d'
        self.position = normOpt['position'] if normOpt else None

        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt

        # 1. actnorm
        self.actnorm = models.modules.FlowActNorms.ActNorm2d(in_channels, actnorm_scale)
        # print('in_channels',in_channels)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = models.modules.Permutations.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)

        # 3. coupling

        self.affine = models.modules.FlowAffineCouplingsAblation_final.CondAffineSeparatedAndCond_1dflow(in_channels=in_channels,
                                                                                                opt=opt)
           


    def forward(self, input, logdet=None, reverse=False,cond_fea=None):
        if not reverse:
            return self.normal_flow(input, logdet,cond_fea)
        else:
            return self.reverse_flow(input, logdet,cond_fea)

    def normal_flow(self, z, logdet,cond_fea):
        # print(z.max())
        # 1. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)

        need_features = self.affine_need_features()

        # 3. coupling

        z, logdet = self.affine(input=z, logdet=logdet, reverse=False,cond_fea=cond_fea)

        return z, logdet

    def reverse_flow(self, z, logdet,cond_fea):

        need_features = self.affine_need_features()

        # 1.coupling

        z, logdet = self.affine(input=z, logdet=logdet, reverse=True,cond_fea=cond_fea)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        # print(z.max())
        return z, logdet

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features

class FlowStepnocond(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "squeeze_invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_alternating_2_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlign": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1SubblocksShuf": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder4": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive",
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None):
        # check configures
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector

        self.norm_type = normOpt['type'] if normOpt else 'ActNorm2d'
        self.position = normOpt['position'] if normOpt else None

        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt

        # 1. actnorm
        self.actnorm = models.modules.FlowActNorms.ActNorm2d(in_channels, actnorm_scale)
        # print('in_channels',in_channels)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = models.modules.Permutations.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)

        # 3. coupling

        self.affine = models.modules.FlowAffineCouplingsAblation_final.CondAffineSeparated_1dflow(in_channels=in_channels,
                                                                                                opt=opt)
           


    def forward(self, input, logdet=None, reverse=False,cond_fea=None):
        if not reverse:
            return self.normal_flow(input, logdet,cond_fea)
        else:
            return self.reverse_flow(input, logdet,cond_fea)

    def normal_flow(self, z, logdet,cond_fea):
        # print(z.max())
        # 1. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)

        need_features = self.affine_need_features()

        # 3. coupling

        z, logdet = self.affine(input=z, logdet=logdet, reverse=False,cond_fea=cond_fea)

        return z, logdet

    def reverse_flow(self, z, logdet,cond_fea):

        need_features = self.affine_need_features()

        # 1.coupling

        z, logdet = self.affine(input=z, logdet=logdet, reverse=True,cond_fea=cond_fea)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        # print(z.max())
        return z, logdet

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features
