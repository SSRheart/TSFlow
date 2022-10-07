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

import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch

import pickle
import imageio

class ONEDataset(data.Dataset):
    def __init__(self, opt,train=False):
        super(ONEDataset, self).__init__()
        self.opt = opt
        self.crop_size = opt.get("GT_size")
        self.scale = None
        self.maskkey = opt.get("local")
        self.random_scale_list = [1]
        self.key = opt.get('phase')
        self.root = '/opt/data/private/Data'
        self.mask_dir = self.root + '/CSRNET/MASK_resize/'
        self.gtname = opt.get('name')
        # print(self.gtname)
        if self.key=='train':

            self.low_dir1 = opt.get("dataroot_LQ")
            self.high_dirc = opt.get("dataroot_GT")
            # self.low_dir1 = self.low_dir1.replace('/task/Data/','/home/oem/Tasks/Datasets/')
            # self.high_dirc = self.high_dirc.replace('/task/Data/','/home/oem/Tasks/Datasets/')

            self.index = [i[0:-4] for i in sorted(os.listdir(self.high_dirc))]
            self.nameslow = [self.low_dir1 +'/'+ index + '.jpg' for index in self.index]
            self.nameshigh =[self.high_dirc +'/'+ index + '.jpg' for index in self.index]
            # self.nameslow = self.nameslow[1500*2:1500*3]
            # self.nameshigh = self.nameshigh[1500*2:1500*3]


        elif self.key=='val':

            self.low_dir1 = opt.get("dataroot_LQ")
            self.high_dirc = opt.get("dataroot_GT")
            # self.low_dir1 = self.low_dir1.replace('/task/Data/','/home/oem/Tasks/Datasets/')
            # self.high_dirc = self.high_dirc.replace('/task/Data/','/home/oem/Tasks/Datasets/')

                
            self.index = [i[0:-4] for i in sorted(os.listdir(self.high_dirc))]

            self.nameslow = [self.low_dir1 +'/'+ index + '.jpg' for index in self.index]
            self.nameshigh = [self.high_dirc +'/'+ index + '.jpg' for index in self.index]


            self.LOWTOTAL=[]
            self.HIGHTOTAL=[]


            for i in range(len(self.index)):


                hr = imageio.imread(self.nameshigh[i]).transpose(2,0,1)#.astype(np.uint8)
                lr = imageio.imread(self.nameslow[i]).transpose(2,0,1)#.astype(np.uint16)

                self.LOWTOTAL.append(lr)
                self.HIGHTOTAL.append(hr)
        elif self.key=='test':

            self.low_dir1 = opt.get("dataroot_LQ")
            self.high_dirc = opt.get("dataroot_GT")
            # self.low_dir1 = self.low_dir1.replace('/task/Data/','/home/oem/Tasks/Datasets/')
            # self.high_dirc = self.high_dirc.replace('/task/Data/','/home/oem/Tasks/Datasets/')

                
            self.index = [i[0:-4] for i in sorted(os.listdir(self.high_dirc))]

            self.nameslow = [self.low_dir1 +'/'+ index + '.jpg' for index in self.index]
            self.nameshigh = [self.high_dirc +'/'+ index + '.jpg' for index in self.index]


            self.LOWTOTAL=[]
            self.HIGHTOTAL=[]


            for i in range(len(self.index)):


                hr = imageio.imread(self.nameshigh[i]).transpose(2,0,1)#.astype(np.uint8)
                lr = imageio.imread(self.nameslow[i]).transpose(2,0,1)#.astype(np.uint16)

                self.LOWTOTAL.append(lr)
                self.HIGHTOTAL.append(hr)
        gpu = True
        augment = True

        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = True #opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)


        # n_max = opt["n_max"] if "n_max" in opt.keys() else int(1e8)

        t = time.time()


        t = time.time() - t


        self.gpu = gpu
        self.augment = augment

        self.measures = None

    def load_pkls(self, path, n_max):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        images = images[:n_max]
        images = [np.transpose(image, [2, 0, 1]) for image in images]
        return images

    def __len__(self):
        return len(self.nameshigh)

    def __getitem__(self, idx):
        # print('experta',self.namesexperta[idx])
        # print('gt',self.nameshigh[idx])
        if self.key=='train':
            # print(self.nameshigh[idx])

            mask_random = np.random.choice(['0','1'])
            hr = imageio.imread(self.nameshigh[idx]).transpose(2,0,1)#.astype(np.uint8)
            lr = imageio.imread(self.nameslow[idx]).transpose(2,0,1)#.astype(np.uint16)


        # self.label = self.labels[idx]

            # hr, lr = random_crop(hr, lr, self.crop_size)
        else:

            hr = self.HIGHTOTAL[idx]
            lr = self.LOWTOTAL[idx]


        hr = hr / 255.0
        lr = lr / 255.0
        # print('input',lr.shape)


        hr = torch.Tensor(hr)
        lr = torch.Tensor(lr)

        return {'LQ': lr, 'GT': hr, 'LQ_path': str(idx), 'GT_path': str(idx)}

    def print_and_reset(self, tag):
        m = self.measures
        kvs = []
        for k in sorted(m.keys()):
            kvs.append("{}={:.2f}".format(k, m[k]))
        print("[KPI] " + tag + ": " + ", ".join(kvs))
        self.measures = None


def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 2).copy()
    seg = seg if random_choice else np.flip(seg, 2).copy()
    return img, seg


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(1, 2)).copy()
    seg = np.rot90(seg, random_choice, axes=(1, 2)).copy()
    return img, seg


def random_crop(hr, lr, size_hr):
    # size_lr = size_hr // scale
    size_lr = size_hr
    size_lr_x = lr.shape[1]
    size_lr_y = lr.shape[2]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[:, start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr]

    # HR Patch
    start_x_hr = start_x_lr * 1
    start_y_hr = start_y_lr * 1
    hr_patch = hr[:, start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr]


    return hr_patch, lr_patch


def center_crop(img, size):
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, border:-border, border:-border]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]

