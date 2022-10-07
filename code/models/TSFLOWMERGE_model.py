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
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import logging
from collections import OrderedDict
from utils.util import get_resume_paths, opt_get

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
import imageio
import numpy as np
logger = logging.getLogger('base')

import torch
class TSFLOWMERGEModel(BaseModel):
    def __init__(self, opt, step):
        super(TSFLOWMERGEModel, self).__init__(opt)
        self.opt = opt

        self.heats = opt['val']['heats']
        self.n_sample = opt['val']['n_sample']
        self.hr_size = opt_get(opt, ['datasets', 'train', 'center_crop_hr_size'])
        self.hr_size = 160 if self.hr_size is None else self.hr_size
        self.lr_size = self.hr_size // opt['scale']

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 1  # non dist training

            # self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt, step).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()

        if opt_get(opt, ['path', 'resume_state'], 1) is not None:
            self.load()
        else:
            print("WARNING: skipping initial loading, due to resume_state None")
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)

        if self.is_train:
            self.netG.train()

            self.init_optimizer_and_scheduler(train_opt)
            self.log_dict = OrderedDict()

    def to(self, device):
        self.device = device
        self.netG.to(device)

    def init_optimizer_and_scheduler(self, train_opt):


        self.optimizers = []
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        optim_params_styleencoder = []
        optim_params_StyleFLow = []

        optim_params_retouchnet = []

        optim_params_pretrainencoder = []



        
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            # print(k, v.requires_grad)
            if v.requires_grad:
                if '.styleencoder' in k:
                    optim_params_styleencoder.append(v)

                elif '.retouchnet.' in k:

                    optim_params_retouchnet.append(v)

                    
                elif '.StyleFLow1.' in k:
                    optim_params_StyleFLow.append(v)
                    # print(k)
                elif '.condflow1' in k:
                    optim_params_StyleFLow.append(v)
                    # print(k)

        self.optimizer_F = torch.optim.Adam(
            [
                {"params": optim_params_StyleFLow, "lr": train_opt['lr_G'],
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},
            ],
        )
        self.optimizer_S = torch.optim.Adam(
            [
                {"params": optim_params_styleencoder, "lr": train_opt['lr_G'],
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},

                {"params": optim_params_retouchnet, "lr": train_opt['lr_G'],
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},

            ],
        )
        self.optimizer_E = torch.optim.Adam(
            [
                {"params": optim_params_pretrainencoder, "lr": train_opt['lr_G'],
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},
            ],
        )



    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ

        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def optimize_parameters(self, step):
        # print(step)
        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_E.zero_grad()
        self.optimizer_S.zero_grad()
        self.optimizer_F.zero_grad()
        losses = {}
        weight_fl = opt_get(self.opt, ['train', 'weight_fl'])
        # index = 0
        weight_fl = 1 #if weight_fl is None else weight_flretain_graph=True
        if weight_fl > 0:
            restored, _ = self.netG(gt=self.real_H, lr=self.var_L, reverse=False,restoredkey=True)
            enhanced_restoreloss = torch.mean(torch.abs(restored[-3]-self.real_H)) + \
            torch.mean(torch.abs(restored[-2]-self.real_H)) + \
            torch.mean(torch.abs(restored[-1]-self.real_H))
            enhanced_restoreloss.backward()
            self.optimizer_S.step()
        

            nll,_ = self.netG(gt=self.real_H, lr=self.var_L, reverse=False,flowkey=True)
            nllloss = torch.mean(nll)#+torch.mean(nllpred)
            nllloss.backward()
            self.optimizer_F.step()


        return enhanced_restoreloss,nllloss#torch.mean(nll)


    def test(self):
        self.netG.eval()
        self.fake_H = {}

        with torch.no_grad():
            self.fake_H = self.netG(gt=self.real_H, lr=self.var_L, reverse=True)
        self.netG.train()
        return nll.mean().item()

    def get_encode_nll(self, lq, gt):
        self.netG.eval()
        with torch.no_grad():
            _, nll, _ = self.netG(gt=gt, lr=lq, reverse=False)
        self.netG.train()
        return nll.mean().item()

    def get_sr(self, hq,lq, heat=None, seed=None, z=None, epses=None):
        return self.get_sr_with_z(hq,lq, heat, seed, z, epses)[0],self.get_sr_with_z(hq,lq, heat, seed, z, epses)[2]
    def expert(self,hq,lq, heat=None, seed=None, epses=None,z_1d=None,z_2d=None):
        _,_,preenhanced = self.netG(gt=hq, lr=lq, reverse=True,sample=True)
        return preenhanced
    def get_unpair(self,hq,lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()
        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z

        with torch.no_grad():
            sr, _,logdet,interout = self.netG(gt=hq,lr=lq, z=z, eps_std=heat, reverse=True, epses=epses)
        self.netG.train()
        return sr, z
    def test_sample(self,hq,lq, heat=None, seed=None, epses=None,z_1d=None,z_2d=None,restoredkey=False,latentkey=False,flowkey=False):
        self.netG.eval()
        # z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z
        # # print('z_1d',z_1d.shape)
        # with torch.no_grad():
        #     if flowkey==False:

        #         enhanced, sr = self.netG(gt=hq,lr=lq, reverse=False,z_1d=z_1d,test=True,restoredkey=restoredkey,latentkey=latentkey,flowkey=flowkey)
        #         self.netG.train()
        #         return enhanced,sr

        #     elif flowkey==True:
        #         enhanced, condifeas = self.netG(gt=hq,lr=lq, reverse=False,z_1d=z_1d,test=True,restoredkey=restoredkey,latentkey=latentkey,flowkey=flowkey)

        #         self.netG.train()
        #         return enhanced,condifeas
        # print('z_1d',z_1d.shape)
        with torch.no_grad():
            # if flowkey==False:

            enhanced, sr = self.netG(gt=hq,lr=lq, reverse=False,z_1d=z_1d,test=True,restoredkey=restoredkey,latentkey=latentkey,flowkey=flowkey)
            self.netG.train()
            return enhanced,sr

            # elif flowkey==True:
            #     enhanced, condifeas = self.netG(gt=hq,lr=lq, reverse=False,z_1d=z_1d,test=True,restoredkey=restoredkey,latentkey=latentkey,flowkey=flowkey)

            #     self.netG.train()
            #     return enhanced,condifeas
    def get_train_z(self,hq,lq, heat=None, seed=None, epses=None,z_1d=None,z_2d=None):
        self.netG.eval()
        # z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z
        # print('z_1d',z_1d.shape)
        with torch.no_grad():
            z, _ = self.netG(gt=hq,lr=lq, reverse=False)
        self.netG.train()
        return z

    def get_encode_z(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, _, _ ,_= self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z

    def get_encode_z_and_nll(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z, nll

    def get_results(self, lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()

        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z

        with torch.no_grad():
            sr, _,logdet,interout = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses)
        self.netG.train()
        return sr, z,interout
    def get_sr_with_z(self, hq,lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()
        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z

        with torch.no_grad():
            sr, _,logdet,interout = self.netG(gt=hq,lr=lq, z=z, eps_std=heat, reverse=True, epses=epses)
        self.netG.train()
        return sr, z,interout

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        if seed: torch.manual_seed(seed)


        C = 3 #self.netG.module.flowUpsamplerNet.C
        H = lr_shape[2]# // self.netG.module.flowUpsamplerNet.scaleH)
        W = lr_shape[3]# // self.netG.module.flowUpsamplerNet.scaleW)
        z = torch.normal(mean=0, std=heat, size=(batch_size, C, H, W)) if heat > 0 else torch.zeros(
            (batch_size, C, H, W))

        return z

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        for heat in self.heats:
            for i in range(self.n_sample):
                out_dict[('SR', heat, i)] = self.fake_H[(heat, i)].detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        _, get_resume_model_path = get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)
            return

        load_path_G = self.opt['path']['pretrain_model_G']
        load_submodule = self.opt['path']['load_submodule'] if 'load_submodule' in self.opt['path'].keys() else 'InvNet'

        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True),
                              submodule=load_submodule)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
