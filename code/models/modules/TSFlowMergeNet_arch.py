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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.BackboneNet_arch import StyleEncoder,RetouchNet,TSFlow_encoder
from models.modules.Flow1dNet import Flow1dNet
import models.modules.thops as thops
import models.modules.flow as flow
from utils.util import opt_get

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) 
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class TSFlowMergeNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None, opt=None, step=None):
        super(TSFlowMergeNet, self).__init__()

        self.opt = opt
        self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
                            None else opt_get(opt, ['datasets', 'train', 'quant'])

        # self.gennet = Genseed()
        self.styleencoder1 = StyleEncoder(nf=32)
        self.styleencoder2 = StyleEncoder(nf=32)
        self.styleencoder3 = StyleEncoder(nf=32)
        self.retouchnet = RetouchNet()

        self.condflow1 = TSFlow_encoder()

        train_InvNet_delay = opt_get(self.opt, ['network_G', 'train_InvNet_delay'])
        self.StyleFLow1 = Flow1dNet((1, 1, 32), 32, 8,
                             flow_coupling=opt['network_G']['flow']['coupling'], opt=opt)
        self.i = 0

    def set_zerodce_training(self, trainable):
        if self.zerodce_training != trainable:
            for p in self.zerodce.parameters():
                p.requires_grad = trainable
            self.zerodce_training = trainable
            return True
        return False
    def set_GNet_training(self, trainable):
        if self.Gencode_training != trainable:
            for p in self.Gencode.parameters():
                p.requires_grad = trainable
            self.Gencode_training = trainable
            return True
        return False

    def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, epses=None, reverse_with_grad=False,
                lr_enc=None,
                add_gt_noise=False, step=None, y_label=None,z_1d=None,test=False,sample=None,expert=None,restoredkey=False,latentkey=False,flowkey=False):

        if not reverse and test==False:

            return self.normal_flow(gt, lr, lr_enc=lr_enc, add_gt_noise=add_gt_noise,
                                    restoredkey=restoredkey,latentkey=latentkey,flowkey=flowkey)
        if test:
            return self.test_flow(gt , lr , lr_enc=lr_enc,add_gt_noise=add_gt_noise,
                                z_1d=z_1d,restoredkey=restoredkey,latentkey=latentkey,flowkey=flowkey)



    def histogram_layer(self,img,max_bin):
        relu = nn.ReLU(inplace=True)

        tmp_list = []
        for i in range(max_bin + 1):
            histo = relu(1-torch.abs(img - i / float(max_bin)) * float(max_bin))
            tmp_list.append(histo)
        histogram_tensor = torch.cat(tmp_list, 1)
        return histogram_tensor
    def normal_flow(self, gt, lr, y_onehot=None, epses=None, lr_enc=None, add_gt_noise=True,restoredkey=False,latentkey=False,flowkey=False):

        if restoredkey==True:
            fea1 = self.styleencoder1(lr, gt)
            out1 = self.retouchnet(lr,fea1)
            fea2 = self.styleencoder2(out1, gt)
            out2 = self.retouchnet(lr,fea1+fea2)
            fea3 = self.styleencoder3(out2, gt)
            out3 = self.retouchnet(lr,fea1+fea2+fea3)
            return [out1,out2,out3],fea1+fea2+fea3


        if flowkey==True:
            fea1 = self.styleencoder1(lr, gt)
            out1 = self.retouchnet(lr,fea1)
            fea2 = self.styleencoder2(out1, gt)
            out2 = self.retouchnet(lr,fea1+fea2)
            fea3 = self.styleencoder3(out2, gt)
            out3 = self.retouchnet(lr,fea1+fea2+fea3)
            globalfeas = fea1+fea2+fea3



            lr_feas = self.condflow1(lr,lr,key='test')[:,:,None,None]
            globalfeas = globalfeas[:,:,None,None].detach()
            logdet1d = torch.zeros_like(globalfeas[:, 0, 0, 0])
            epses1d, logdet1d = self.StyleFLow1(gt=globalfeas, logdet=logdet1d, reverse=False, epses=epses,
                                                cond_fea=lr_feas)
            objective1d = logdet1d.clone()

            objective1d = objective1d + flow.GaussianDiag.logp(None, None, epses1d)
            pixels1d = thops.pixels(epses1d)
            nll1d = (-objective1d) / float(np.log(2.) * pixels1d)
            return nll1d,epses1d
    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real


    def test_flow(self, gt, lr, epses=None, lr_enc=None, add_gt_noise=True,z_1d=None,restoredkey=False,latentkey=False,flowkey=False):
        if restoredkey==True:
            fea1 = self.styleencoder1(lr, gt)
            out1 = self.retouchnet(lr,fea1)
            fea2 = self.styleencoder2(out1, gt)
            out2 = self.retouchnet(lr,fea1+fea2)
            fea3 = self.styleencoder3(out2, gt)
            out3 = self.retouchnet(lr,fea1+fea2+fea3)
            
            return out3,fea1+fea2+fea3


        if flowkey==True:

            lr_feas = self.condflow1(lr,lr,key='test')[:,:,None,None]

            logdet1d = torch.zeros_like(lr[:, 0, 0, 0])
            epses1d, logdet1d = self.StyleFLow1(gt=z_1d, z=z_1d, logdet=logdet1d, reverse=True, epses=epses,
                                            cond_fea=lr_feas)

            outs = self.retouchnet(lr,epses1d[:,:,0,0])

            return outs,z_1d
        if latentkey==True:
            fea1 = self.styleencoder1(lr, gt)
            out1 = self.retouchnet(lr,fea1)
            fea2 = self.styleencoder2(out1, gt)
            out2 = self.retouchnet(lr,fea1+fea2)
            fea3 = self.styleencoder3(out2, gt)
            out3 = self.retouchnet(lr,fea1+fea2+fea3)
            globalfeas = fea1+fea2+fea3
            lr_feas = self.condflow1(lr,lr,key='test')[:,:,None,None]

            globalfeas = globalfeas[:,:,None,None].detach()
            logdet1d = torch.zeros_like(globalfeas[:, 0, 0, 0])

            epses1d, logdet1d = self.StyleFLow1(gt=globalfeas, logdet=logdet1d, reverse=False, epses=epses,
                                                cond_fea=lr_feas)

            return epses1d,epses1d
