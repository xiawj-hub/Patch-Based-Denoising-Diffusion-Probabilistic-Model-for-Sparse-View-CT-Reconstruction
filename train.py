import torch
import torch.nn as nn
import argparse
import os
import glob
import json
import numpy as np
import time
import datetime
import functools
from model.ddpm_modules import UNet
from model.sde import SDEDiffusion
from model.sde import sde_lib
from model import ema
from data.dataset import MayoDataset
from utils.metrics import calculate_psnr, calculate_ssim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.nn import init

from torch.utils.tensorboard import SummaryWriter

class patch_operator():
    def __init__(self, img_size, patch_size, step, padding=1):
        self.img_size = img_size
        self.patch_size = patch_size
        self.step = step
        self.patch_mask = 0
        self.w = list(range(0, img_size[0] - patch_size[0], step))
        if img_size[0] - patch_size[0] not in self.w:
            self.w.append(img_size[0] - patch_size[0])
        self.h = list(range(0, img_size[1] - patch_size[1], step))
        if img_size[1] - patch_size[1] not in self.h:
            self.h.append(img_size[1] - patch_size[1])

        self.patch_mask = torch.ones(patch_size[0] - padding*2, patch_size[1] - padding*2).cuda()
        self.patch_mask = torch.nn.functional.pad(self.patch_mask, [padding]*4)
        self.mask = self.get_mask()
        self.padding = padding

    def mask_padding(self, i, j):
        mask = self.patch_mask.clone()
        if i == 0:
            mask[0, :] = 1.0
        elif i == self.w[-1]:
            mask[-1,:] = 1.0

        if j == 0:
           mask[:, 0] = 1.0
        elif j == self.h[-1]:
            mask[:,-1] = 1.0 

        return mask
    
    def to_patch(self, img):
        b = img.shape[0]
        n = len(self.w) * len(self.h)
        patchset = torch.zeros(n * b, 1, *self.patch_size, device = img.device)
        for idx_i, i in enumerate(self.w):
            for idx_j, j in enumerate(self.h):
                idx = idx_i * len(self.h) + idx_j
                patch = img[:,:,i:i+self.patch_size[0],j:j+self.patch_size[1]]
                patchset[idx::n] = patch
        return patchset


    def to_img(self, patchset):
        n = len(self.w) * len(self.h)
        b = patchset.shape[0] // n
        img = torch.zeros(b, 1, *self.img_size, device = patchset.device)
        for idx_i, i in enumerate(self.w):
            for idx_j, j in enumerate(self.h):
                idx = idx_i * len(self.h) + idx_j
                mask_padding = self.mask_padding(i,j)
                patch = patchset[idx::n] * mask_padding
                img[:,:,i:i+self.patch_size[0],j:j+self.patch_size[1]] += patch
        img = img / self.mask
        return img

    def get_mask(self):
        n = len(self.w) * len(self.h)
        patchset = torch.ones(n, 1, *self.patch_size, device = self.patch_mask.device)
        img = torch.zeros(1, 1, *self.img_size, device = patchset.device)
        for idx_i, i in enumerate(self.w):
            for idx_j, j in enumerate(self.h):
                idx = idx_i * len(self.h) + idx_j
                mask_padding = self.mask_padding(i,j)
                patch = patchset[idx] * mask_padding
                img[:,:,i:i+self.patch_size[0],j:j+self.patch_size[1]] += patch
        return img

    def random_sample(self, img, k):
        b = img.shape[0]
        w = torch.randperm(self.w[-1]+1)[:k]
        h = torch.randperm(self.h[-1]+1)[:k]
        n = k * k
        patchset = torch.zeros(n * b, 1, *self.patch_size, device = img.device)
        for idx_i, i in enumerate(w):
            for idx_j, j in enumerate(h):
                idx = idx_i * len(h) + idx_j
                patch = img[:,:,i:i+self.patch_size[0],j:j+self.patch_size[1]]
                patchset[idx::n] = patch
        return patchset

def setup_parser(arguments, title):
    parser = argparse.ArgumentParser(description=title,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key, val in arguments.items():
        parser.add_argument('--%s' % key,
                            type=eval(val["type"]),
                            help=val["help"],
                            default=val["default"],
                            nargs=val["nargs"] if "nargs" in val else None)
    return parser

def get_parameters(config_file, title=None):
    with open(config_file) as data_file:
        data = json.load(data_file)
    parser = setup_parser(data, title)
    parameters = parser.parse_args()
    return parameters


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
 
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))

def worker(rank, args):
    dist.init_process_group("nccl", rank=rank, world_size=args.ngpus)
    torch.cuda.set_device(rank)
    
    net = UNet(
        in_channel=args.in_ch,
        out_channel=args.out_ch,
        norm_groups=args.gp_norm,
        inner_channel=args.inner_ch,
        channel_mults=args.ch_mul,
        attn_res=args.attn_res,
        res_blocks=args.res_blocks,
        dropout=args.dropout,
        image_size=args.image_size
    )    

    net = net.cuda()
    init_weights(net, init_type='orthogonal')
    ema_fn = ema.ExponentialMovingAverage(net.parameters(), args.ema_decay)

    if not os.path.exists(args.checkpointdir) and rank == 0:
        os.mkdir(args.checkpointdir)
    dist.barrier()

    ckpt_list = glob.glob(os.path.join(args.checkpointdir, 'Model_*.pth'))

    current_epoch = 0
    current_iter = 0 
    cost = 0
    if not len(ckpt_list) == 0:
        ckpt_list.sort()
        ckpt = ckpt_list[-1]
        state_dict = torch.load(ckpt, map_location={"cuda:0": f"cuda:{rank}"})
        net.load_state_dict(state_dict['model'])
        ema_fn.load_state_dict(state_dict['ema'])
        current_epoch = state_dict['epoch']
        current_iter = state_dict['iter'] + 1
        cost = state_dict['cost']

    if args.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, N=args.num_scales)
        sampling_eps = 1e-3
    elif args.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=args.beta_min, beta_max=args.beta_max, N=args.num_scales)
        sampling_eps = 1e-3
    elif args.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=args.sigma_min, sigma_max=args.sigma_max, N=args.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {args.sde} unknown.")

    diffusion = SDEDiffusion(
        net,
        sde,
        args.snr,
        time_eps = sampling_eps,
        predictor = args.predictor,
        corrector = args.corrector
    )

    diffusion = DistributedDataParallel(diffusion, device_ids=[rank])

    train_dataset = MayoDataset(args.dataroot, 'train')
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.ncpus, sampler=train_sampler)

    test_dataset = MayoDataset(args.dataroot, 'test')
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.ncpus, sampler=test_sampler)
    
    if rank == 0:
        writer = SummaryWriter(args.logdir)

    opt = torch.optim.Adam(diffusion.parameters(), lr=args.lr)
    patch_transform = patch_operator([736,736], [64, 64], 32)
    
    time_start = time.time()
    while current_iter < args.n_iter:        
        train_sampler.set_epoch(current_epoch)
        for _, train_data in enumerate(train_loader):
            x = train_data['label']
            patchset = patch_transform.random_sample(x, 9)
            x = patchset
            x = x.cuda()
            opt.zero_grad()
            loss = diffusion(x)
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.module.parameters(), args.grad_clip)
            opt.step()
            ema_fn.update(diffusion.module.model.parameters())           
            
            if rank == 0:
                if (current_iter + 1) % 10 == 0:
                    writer.add_scalar('train_loss', loss.item(), current_iter)
                    print("[Epoch: {:04d}] [Iter: {:07d}] [loss: {:.4e}]".format(current_epoch, current_iter, loss.item()))
                if (current_iter + 1) % args.save_checkpoint_freq == 0:
                    time_end = time.time()
                    cost += round(time_end-time_start)
                    checkpoint = {'model':diffusion.module.model.state_dict(),
                                    'ema':ema_fn.state_dict(),
                                    'epoch':current_epoch,
                                    'iter':current_iter,
                                    'cost':cost}
                    torch.save(checkpoint, os.path.join(args.checkpointdir, 'Model_{:03d}.pth'.format((current_iter + 1) // args.save_checkpoint_freq)))
                    time_start = time_end
            dist.barrier()
            # if (current_iter + 1) % args.val_freq == 0:
            #     ema_fn.store(diffusion.module.model.parameters())
            #     ema_fn.copy_to(diffusion.module.model.parameters())
            #     diffusion.eval()
            #     val_loss = 0
            #     for _, test_data in enumerate(test_loader):
            #         x = test_data['label']
            #         patchset = patch_transform.to_patch(x)
            #         x = patchset
            #         x = x.cuda()

            #         with torch.no_grad():
            #             loss = diffusion(x)
            #             val_loss = val_loss + loss
                    
            #     ema_fn.restore(diffusion.module.model.parameters())
            #     diffusion.train()
            #     dist.all_reduce(val_loss)
            #     val_loss = val_loss / (len(test_loader) * args.ngpus)
            #     if rank == 0:
            #         writer.add_scalar('val_loss', val_loss.item(), current_iter)
            current_iter += 1
        current_epoch += 1


if __name__ == "__main__":
    config_file = "./configs/vpsde_512_prj_patch_config.json"
    args = get_parameters(config_file)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    mp.spawn(worker, nprocs=args.ngpus, args=(args,))