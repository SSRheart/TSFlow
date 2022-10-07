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


import glob
import sys
from collections import OrderedDict
import torch.nn.functional as F
import itertools
import options.options as option
from Measure import Measure, psnr
from imresize import imresize
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2
import imageio
import math
from utils import util
from tqdm import tqdm
import lpips
from pytorch_msssim import ssim
# from skimage.metrics import structural_similarity as ssim

def write_img(filename, img):
	img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
	cv2.imwrite(filename, img)




def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    print('model_path',model_path)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get("SR"))


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255
def tint16(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 65535


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]

def imwritefea(path,fea):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, fea)
def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]
class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def main():
    ToName = {0:'fea_1',
                   1:'fea_2',
                   2:'fea_3',
                   3:'fea_4',
                   4:'fea_5',
                   5:'fea_6',
                   6:'fea_7',
                   7:'fea_8'
                   }
    conf_path = sys.argv[1]
    conf = conf_path.split('/')[-1].replace('.yml', '')
    model, opt = load_model(conf_path)


    norm_dir = os.path.join('/home/oem/Tasks/Datasets/Starenhancer/test/06-Input-ExpertC1.5/')
    high_dir = os.path.join('/home/oem/Tasks/Datasets/Starenhancer/test/03-Experts-C/')
    namesgt = []
    namesinput = []
    for i in sorted(os.listdir(high_dir)):
        namesgt.append(high_dir+i)
        namesinput.append(norm_dir+i[0:-4]+'.jpg')


    lr_paths = namesinput
    hr_paths = namesgt


    this_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = this_dir#os.path.join(this_dir, '..', 'results', conf)
    print(f"Out dir: {this_dir}")
    print('len(nameshigh)',len(hr_paths))
    fname = f'measure_full.csv'
    fname_tmp = fname + "_"
    path_out_measures = os.path.join(test_dir, fname_tmp)
    path_out_measures_final = os.path.join(test_dir, fname)

    if os.path.isfile(path_out_measures_final):
        df = pd.read_csv(path_out_measures_final)
    elif os.path.isfile(path_out_measures):
        df = pd.read_csv(path_out_measures)
    else:
        df = None


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seeds = torch.from_numpy(np.load('expert_c_latent.npy')).float()#.cuda()

    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPS = AverageMeter()
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.cuda()
    for lr_path, hr_path, idx_test in tqdm(zip(lr_paths, hr_paths, range(len(hr_paths)))):
        lr = cv2.imread(lr_path)[:,:,::-1]
        hr = cv2.imread(hr_path)[:,:,::-1]

        h, w, c = lr.shape
        lr_t = t(lr).to(device)
        hr_t = t(hr).to(device)


        output,_ = model.test_sample(hq=hr_t,lq=lr_t, z_1d=seeds,flowkey=True)
        output = (output.clamp_(0, 1) * 255).round_() / 255.
        mse_loss = F.mse_loss(output, hr_t, reduction='none').mean((1, 2, 3))
        psnr_val = 10 * torch.log10(1 / mse_loss).mean().item()
        _, _, H, W = output.size()
        down_ratio = max(1, round(min(H, W) / 256))

        ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
        					F.adaptive_avg_pool2d(hr_t, (int(H / down_ratio), int(W / down_ratio))), 
        					data_range=1, size_average=False).item()	


        lpips_val = loss_fn_alex(output * 2 - 1, hr_t * 2 - 1).item()		# Richard Zhang
        PSNR.update(psnr_val)
        SSIM.update(ssim_val)
        LPIPS.update(lpips_val)
    print(PSNR.avg,SSIM.avg,LPIPS.avg)

def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).copy()



def printmetrics(metrics,N):
    a = [i for i in range(100)]
    bestindexs  = 0
    best = 0
    for indexs in itertools.combinations(a,N):

        newmetrics= []
        for index in indexs:
            # print(meanpsnr[c[(-1)*i]])
            newmetrics.append(metrics[:,0,index])
        newmetrics = np.array(newmetrics)
        psnr = np.max(newmetrics,0).mean()
        if psnr>best:
            best = psnr
            bestindexs = indexs
            # print(best)
    return best,bestindexs
def printfinalmetrics(metrics):
    print(np.max(metrics[:,0,:],1).shape)
    psnr = np.max(metrics[:,0,:],1).mean()
    ssim = np.max(metrics[:,1,:],1).mean()
    lpips = np.min(metrics[:,2,:],1).mean()

            # print(best)
    return [psnr,ssim,lpips]
def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.2f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


if __name__ == "__main__":
    main()
