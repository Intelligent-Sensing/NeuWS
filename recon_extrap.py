# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import os, time, math, imageio, tqdm, argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

import torch
print(f"Using PyTorch Version: {torch.__version__}")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift
from networks import *
from utils import *
from dataset import *

DEVICE = 'cuda'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--data_dir', default='resized', type=str)
    parser.add_argument('--scene_name', default='0609', type=str)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_t', help='Number of training frames', default=100, type=int)
    parser.add_argument('--num_extrap_t', help='Number of extrapolated frames', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--init_lr', default=1e-3, type=float)
    #parser.add_argument('--final_lr', default=1e-6, type=float)
    parser.add_argument('--silence_tqdm', action='store_true')
    parser.add_argument('--save_per_frame', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--max_intensity', default=30000, type=float)
    parser.add_argument('--save_gif', action='store_true')

    args = parser.parse_args()
    PSF_size = args.width

    # Set to folder with test image
    data_dir = f'{args.root_dir}/{args.data_dir}'
    vis_dir = f'{args.root_dir}/vis/{args.scene_name}'
    os.makedirs(f'{args.root_dir}/vis', exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(f'{vis_dir}/final', exist_ok=True)
    os.makedirs(f'{vis_dir}/final/per_frame', exist_ok=True)

    print(f'Saving output at: {vis_dir}')
    dset = BatchDataset(data_dir, num=args.num_t, max_intensity=args.max_intensity, save_gif=args.save_gif)
    if args.save_gif:
        exit()
    train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    net = TemporalZernNet(width=args.width, PSF_size=PSF_size)
    net = net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr)

    a_slm = np.ones((144, 256))
    a_slm = np.lib.pad(a_slm, (((256 - 144) // 2, (256 - 144) // 2), (0, 0)), 'constant', constant_values=(0, 0))
    a_slm = torch.from_numpy(a_slm).type(torch.float).to(DEVICE)

    total_it = 0
    t = tqdm.trange(args.num_epochs, disable=args.silence_tqdm)

    ############
    # Training loop
    for epoch in t:
        for it, (x_batch, y_batch, idx) in enumerate(train_loader):
            x_batch, y_batch, idx = x_batch.to(DEVICE), y_batch.to(DEVICE), idx.to(DEVICE)
            optimizer.zero_grad()
            cur_t = idx / args.num_t - 0.5
            y, _kernel, sim_g, sim_phs, I_est = net(x_batch, cur_t)
            mse_loss = F.mse_loss(y, y_batch)
            loss = mse_loss
            loss.backward()
            psnr = 10 * -torch.log10(loss).item()
            t.set_postfix(MSE = f'{mse_loss.item():.4e}', PSNR = f'{psnr:.2f}')
            if args.vis_freq > 0 and (total_it % args.vis_freq) == 0:
                kkernel = fftshift(fft2(a_slm * sim_g, norm="forward"), dim=[-2, -1]).abs() ** 2
                yy = F.conv2d(I_est, kkernel, padding='same').squeeze(0)

                fig, ax = plt.subplots(1, 6, figsize=(48, 8))
                ax[0].imshow(y_batch[0].detach().cpu().squeeze(), cmap='gray')
                ax[0].axis('off')
                ax[0].title.set_text('Real I')
                ax[1].imshow(y[0].detach().cpu().squeeze(), cmap='gray')
                ax[1].axis('off')
                ax[1].title.set_text('Sim I')
                ax[2].imshow(I_est.detach().cpu().squeeze(), cmap='gray')
                ax[2].axis('off')
                ax[2].title.set_text('I_est')
                ax[3].imshow(sim_phs[0].detach().cpu().squeeze(), cmap='gray') # , vmin=-3.15, vmax=3.15
                ax[3].axis('off')
                ax[3].title.set_text(f'Sim Phase at t={idx[0]}')
                ax[4].imshow(_kernel[0].detach().cpu().squeeze(), cmap='gray')
                ax[4].axis('off')
                ax[4].title.set_text('Sim PSF')
                ax[5].imshow(yy[0].squeeze().detach().cpu(), cmap='gray')
                ax[5].axis('off')
                ax[5].title.set_text(f'Sim PSF*I_est at t={idx[0]}')
                plt.savefig(f'{vis_dir}/e_{epoch}_it_{it}.jpg')
                plt.clf()
                sio.savemat(f'{vis_dir}/Sim_Phase.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})
            optimizer.step()
            total_it += 1
    
    ############
    # Export final results
    I_est = net.g_im()
    I_est = np.uint8(torch.clamp(I_est, 0, 1).squeeze().detach().cpu().numpy() * 255)

    out_errs = []
    extrap_errs = []
    for t in range(args.num_t + args.num_extrap_t):
        t_tensor = torch.ones(1, 1, 1, 1).to(DEVICE) * (t / args.num_t) - 0.5
        g_out = net.g_g(torch.cat([net.basis.repeat(len(t_tensor), 1, 1, 1).to(DEVICE), net.t_grid * t_tensor], dim = -1))
        sim_phs = g_out.permute(0, 3, 1, 2)
        est_g = torch.exp(1j * sim_phs[0]).detach().cpu().squeeze().numpy()
        out_errs.append(np.uint8(ang_to_unit(np.angle(est_g)) * 255))
        if t >= args.num_t:
            extrap_errs.append(np.uint8(ang_to_unit(np.angle(est_g)) * 255))
        if args.save_per_frame or t >= args.num_t:
          sio.savemat(f'{vis_dir}/final/per_frame/sim_phase_{t}.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})
    imageio.mimsave(f'{vis_dir}/final/final_aberration.gif', out_errs, fps=30)
    imageio.imsave(f'{vis_dir}/final/final_I_est.png', I_est)
    imageio.mimsave(f'{vis_dir}/final/final_extrap_aberration.gif', extrap_errs, fps=30)

    print("Done")