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
from os.path import basename
import math
import argparse
import random
import logging
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import lpips
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.timer import Timer, TickTock
from utils.util import get_resume_paths
import numpy as np
from tqdm import tqdm
import itertools
def getEnv(name): import os; return True if name in os.environ.keys() else False


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_deviceDistIterSampler(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        resume_state_path, _ = get_resume_paths(opt)

        # distributed resuming: all load into default GPU
        if resume_state_path is None:
            resume_state = None
        else:
            print('resume_state_pathresume_state_pathresume_state_pathresume_state_path',resume_state_path)
            device_id = torch.cuda.current_device()
            resume_state = torch.load(resume_state_path,
                                      map_location=lambda storage, loc: storage.cuda(device_id))
            option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

        # tensorboard logger
        if opt.get('use_tb_logger', False) and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            conf_name = basename(args.opt).replace(".yml", "")
            exp_dir = opt['path']['experiments_root']
            log_dir_train = os.path.join(exp_dir, 'tb', conf_name, 'train')
            log_dir_valid = os.path.join(exp_dir, 'tb', conf_name, 'valid')
            tb_logger_train = SummaryWriter(log_dir=log_dir_train)
            tb_logger_valid = SummaryWriter(log_dir=log_dir_valid)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # print('debugtrain',dataset_opt)
            train_set = create_dataset(dataset_opt)

            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            # print('debugval',dataset_opt)

            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        elif phase == 'test':
            test_set = create_dataset(dataset_opt)
            # print('debugtest',dataset_opt)

            test_loader = create_dataloader(test_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(test_set)))

        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    current_step = 0 if resume_state is None else resume_state['iter']
    model = create_model(opt, current_step)
    loss_fn_alex_1 = lpips.LPIPS(net='alex', version='0.1')#.to(device)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        # model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    timer = Timer()
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    timerData = TickTock()
    for epoch in range(start_epoch, start_epoch + 10000):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        lossres = AverageMeter()
        lossnll = AverageMeter()

        for _, train_data in enumerate(train_loader):
            # timerData.tock()
            current_step += 1
            model.feed_data(train_data)

            # # # # try:
            loss1,loss2 = model.optimize_parameters(current_step)
            lossres.update(loss1.item())
            lossnll.update(loss2.item())

            #### save models and training states
            if current_step % 500 == 0:
                # print('aaaaaaaaaaaaaaaaaaaa')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    with open(os.path.join(opt['path']['root'], "TRAIN_DONE"), 'w') as f:
        f.write("TRAIN_DONE")

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
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
def lpips_norm(img):
	img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
	img = img / (255. / 2.) - 1
	return torch.Tensor(img)#.to(device)

def calc_lpips(x_mask_out, x_canon, loss_fn_alex_1):
	lpips_mask_out = lpips_norm(x_mask_out)
	lpips_canon = lpips_norm(x_canon)
	# LPIPS_0 = loss_fn_alex_0(lpips_mask_out, lpips_canon)
	LPIPS_1 = loss_fn_alex_1(lpips_mask_out, lpips_canon)
	return LPIPS_1.detach().cpu()

def calc_metrics(x_mask_out, x_canon, loss_fn_alex_1):
	# psnr = calc_psnr_np(x_mask_out, x_canon)
	SSIM = ssim(x_mask_out, x_canon, win_size=11, data_range=255, multichannel=True, gaussian_weights=True)
	LPIPS_1 = calc_lpips(x_mask_out, x_canon, loss_fn_alex_1)
	return SSIM, LPIPS_1
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
if __name__ == '__main__':
    main()
