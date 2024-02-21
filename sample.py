import torch
import argparse
import os
import glob
import json
import numpy as np
import time
import datetime
import functools
from model.ddpm_modules import UNet
from model.sde import InvSolver
from model.sde import sde_lib
from model import ema
import ctlib
from data.dataset import MayoDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

def Afun(x):
    return x

def Ainv(y,x):
    return y

def CG(x0, y, options, iter, tol=1e-4):
    # min _x 1/2 * ||Hx-y||_2^2 + alpha / 2||x-f||_2^2
    x = x0.clone()
    for i in range(iter):
        u = x.clone()
        tmp = ctlib.projection(x, options) - y
        grad = ctlib.projection_t(tmp, options)
        if i == 0:
            d = - grad
        else:
            beta = (grad ** 2).sum((2,3), keepdim=True) / (grad_old ** 2).sum((2,3), keepdim=True).clamp_(min=1e-8)
            d = - grad + beta * d
        grad_old = grad
        Hd = ctlib.projection(d, options)
        step = - (grad * d).sum((2,3), keepdim=True) / ((Hd ** 2).sum((2,3), keepdim=True)).clamp_(min=1e-8)
        x = x + step * d
        error = ((x- u) ** 2).sum() / x0.size(0)
        if error < tol:
            break
    return x

class fan_ed():
    def __init__(self, views, dets, width, height, dImg, dDet, dAng, s2r, d2r, binshift):
        self.options = torch.Tensor([views, dets, width, height, dImg, dDet, 0, dAng, s2r, d2r, binshift, 1])
        self.options = self.options.float().cuda()
        self.width = width
        self.height = height

    def Afun(self, image):
        return ctlib.projection(image.contiguous(), self.options)

    def Ainv(self, y, x):
        if x is None:
            x = torch.zeros(y.shape[0], 1, self.width, self.height, device=y.device)
        x = CG(x, y, self.options, 5)
        return x

    def fbp(self, proj):
        return ctlib.fbp(proj.contiguous(), self.options)

class patch_operator():
    def __init__(self, img_size, patch_size, step):
        self.img_size = img_size
        self.patch_size = patch_size
        self.step = step
        divisor, l = self.get_divisor()
        self.divisor = divisor.cuda()
        self.l = l
    
    def to_patch(self, img):
        patchset = torch.nn.functional.unfold(img, self.patch_size, stride=self.step)
        patchset = patchset.permute((0, 2, 1)).reshape(-1, 1, self.patch_size, self.patch_size)
        return patchset.contiguous()


    def to_img(self, patchset):
        patchset = patchset.reshape(-1, self.l, self.patch_size**2).permute((0, 2, 1))
        img = torch.nn.functional.fold(patchset, [self.img_size, self.img_size], self.patch_size, stride=self.step)
        img = img / self.divisor
        return img

    def get_divisor(self):
        img = torch.ones(1, 1, self.img_size, self.img_size)
        patchset = torch.nn.functional.unfold(img, self.patch_size, stride=self.step)
        l = patchset.shape[2]
        divisor = torch.nn.functional.fold(patchset, [self.img_size, self.img_size], self.patch_size, stride=self.step)
        return divisor, l

class transform():
    def __init__(self, patch_operator):
        self.patch_operator = patch_operator
    
    def forward(self, patch):
        img = self.patch_operator.to_img(patch)
        return img

    def backward(self, proj, x):
        patch = self.patch_operator.to_patch(proj)
        return patch

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
    ema_fn = ema.ExponentialMovingAverage(net.parameters(), args.ema_decay)

    result_dir = "./result_sde_prj_patch"

    if not os.path.exists(result_dir) and rank == 0:
        os.mkdir(result_dir)
    dist.barrier()

    ckpt_list = glob.glob(os.path.join(args.checkpointdir, 'Model_*.pth'))  

    if not len(ckpt_list) == 0:
        ckpt_list.sort()
        if args.model_id == -1:
            ckpt = ckpt_list[-1]
        else:
            ckpt = os.path.join(args.checkpointdir, 'Model_{:03d}.pth'.format(args.model_id))
        state_dict = torch.load(ckpt, map_location={"cuda:0": f"cuda:{rank}"})
        net.load_state_dict(state_dict['model'])
        ema_fn.load_state_dict(state_dict['ema'])
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

    projector = fan_ed(736, 736, 512, 512, 0.006641, 0.001184, 0.0085369, 5.95, 4.906, 0)
    projector_sparse = fan_ed(92, 736, 512, 512, 0.006641, 0.001184, 0.0085369*8, 5.95, 4.906, 0)
    patchfy = patch_operator(736, 64, 32)
    mask = torch.ones(736, 736, dtype=torch.float32).cuda() * 0.1
    mask[::8] = 1.0
    mask = mask[None, None,:,:]
    mask = patchfy.to_patch(mask)
    diffusion = InvSolver(
        net,
        sde,
        args.snr,
        Afun,
        Ainv,
        mask,
        args.coeff,
        time_eps = sampling_eps,
        predictor = args.predictor,
        corrector = args.corrector
    )

    diffusion = DistributedDataParallel(diffusion, device_ids=[rank])

    dataset = MayoDataset(args.dataroot, 'test')
    sampler = DistributedSampler(dataset, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=args.ncpus, sampler=sampler)

    ema_fn.store(diffusion.module.model.parameters())
    ema_fn.copy_to(diffusion.module.model.parameters())
    diffusion.eval()
    time_start = time.time()
    conditional = True
    if conditional == False:
        with torch.no_grad():
            out, out_mean = diffusion.module.pc_sampling([1,1,512,512])
            out = out.cpu().numpy()
        for i in range(out.shape[0]):
            np.save(os.path.join(result_dir, 'sample_{}.npy'.format(rank)), out[i].squeeze())

    else:
        for _, test_data in enumerate(test_loader):
            y = test_data['label']
            y = y.cuda()
            y_temp = y[:,:,::8,:]
            x_temp = projector_sparse.fbp(y_temp)
            y_temp = projector.Afun(x_temp)
            y_temp[::8] = y[::8]
            y = y_temp
            y = patchfy.to_patch(y)
            # y = y * mask
            with torch.no_grad():
                out, mean = diffusion.module.pc_sampling(y)
                out = patchfy.to_img(out)
                res = projector.fbp(out)
                out = out.cpu().numpy()
                res = res.cpu().numpy()
            for i in range(out.shape[0]):
                # np.save(os.path.join(result_dir, test_data['path'][i]+'.npy'), out[i].squeeze())
                np.savez(os.path.join(result_dir, test_data['path'][i]+'.npz'), prj=out[i].squeeze(),img=res[i].squeeze())
        time_end = time.time()
        avg_cost = time_end - time_start
        avgcost_tensor = torch.tensor(avg_cost).cuda()
        dist.all_reduce(avgcost_tensor)
        avgcost_tensor = avgcost_tensor / dataset.__len__()
        if rank == 0:
            print("Training cots: {}h{}m{}s".format(cost//3600, (cost%3600) //60, cost%60))
            print("Sampling cost: {}s".format(avgcost_tensor.item()))


if __name__ == "__main__":
    config_file = "./configs/vpsde_512_prj_patch_config.json"
    args = get_parameters(config_file)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    mp.spawn(worker, nprocs=args.ngpus, args=(args,))