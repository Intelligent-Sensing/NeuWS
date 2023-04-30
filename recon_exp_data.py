# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import os, time, math, imageio, tqdm, argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

import torch
print(f"Using PyTorch Version: {torch.__version__}")
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift
from networks import *
from utils import *
from dataset import *

DEVICE = 'cuda'

if __name__ == "__main__":
    """
    python recon_exp_data.py
        --vis_freq -1 --num_t NUM_FRAMES --data_dir DATA_DIR
        --scene_name SCENE_NAME --num_epochs 1000 --batch_size 8
        --max_intensity MAX_RAW_INTENSITY --silence_tqdm"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--data_dir', default='resized', type=str)
    parser.add_argument('--scene_name', default='0609', type=str)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_t', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--init_lr', default=1e-3, type=float)
    parser.add_argument('--final_lr', default=1e-3, type=float)
    parser.add_argument('--silence_tqdm', action='store_true')
    parser.add_argument('--save_per_frame', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_intensity', default=30000, type=float)
    args = parser.parse_args()
    PSF_size = args.width

    ############
    # Setup output folders
    data_dir = f'{args.root_dir}/{args.data_dir}'
    vis_dir = f'{args.root_dir}/vis/{args.scene_name}'
    os.makedirs(f'{args.root_dir}/vis', exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(f'{vis_dir}/final', exist_ok=True)
    print(f'Saving output at: {vis_dir}')
    if args.save_per_frame:
        os.makedirs(f'{vis_dir}/final/per_frame', exist_ok=True)

    ############
    # Training preparations
    dset = BatchDataset(data_dir, num=args.num_t, max_intensity=args.max_intensity)
    x_batches = torch.cat(dset.xs, axis=0).unsqueeze(1).to(DEVICE)
    y_batches = torch.stack(dset.ys, axis=0).to(DEVICE)

    net = TemporalZernNet(width=args.width, PSF_size=PSF_size, use_FFT=True, bsize=args.batch_size)
    net = net.to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.num_epochs // 2, eta_min=args.final_lr)

    total_it = 0
    t = tqdm.trange(args.num_epochs, disable=args.silence_tqdm)

    ############
    # Training loop
    t0 = time.time()
    for epoch in t:
        idxs=torch.randperm(len(dset)).long().to(DEVICE)
        for it in range(0,len(dset),args.batch_size):
            idx = idxs[it:it+args.batch_size]
            x_batch, y_batch = x_batches[idx], y_batches[idx]

            cur_t = idx / args.num_t - 0.5

            optimizer.zero_grad()

            y, _kernel, sim_g, sim_phs, I_est = net(x_batch, cur_t)
            mse_loss = F.mse_loss(y, y_batch)
            loss = mse_loss
            loss.backward()
            optimizer.step()

            t.set_postfix(MSE = f'{mse_loss.item():.4e}')

            if args.vis_freq > 0 and (total_it % args.vis_freq) == 0:
                Abe_est = fftshift(fft2(dset.a_slm.to(DEVICE) * sim_g, norm="forward"), dim=[-2, -1]).abs() ** 2
                yy = F.conv2d(I_est, Abe_est, padding='same').squeeze(0)

                fig, ax = plt.subplots(1, 6, figsize=(48, 8))
                ax[0].imshow(y_batch[0].detach().cpu().squeeze(), cmap='gray')
                ax[0].axis('off')
                ax[0].title.set_text('Real Measurement')
                ax[1].imshow(y[0].detach().cpu().squeeze(), cmap='gray')
                ax[1].axis('off')
                ax[1].title.set_text('Sim Measurement')
                ax[2].imshow(torch.clamp(I_est, 0, 1).detach().cpu().squeeze(), cmap='gray')
                ax[2].axis('off')
                ax[2].title.set_text('I_est')
                ax[3].imshow(sim_phs[0].detach().cpu().squeeze(), cmap='gray') # , vmin=-3.15, vmax=3.15
                ax[3].axis('off')
                ax[3].title.set_text(f'Sim Phase Error at t={idx[0]}')
                ax[4].imshow(_kernel[0].detach().cpu().squeeze(), cmap='gray')
                ax[4].axis('off')
                ax[4].title.set_text('Sim post-SLM PSF')
                ax[5].imshow(yy[0].squeeze().detach().cpu(), cmap='gray')
                ax[5].axis('off')
                ax[5].title.set_text(f'Abe_est * I_est at t={idx[0]}')
                plt.savefig(f'{vis_dir}/e_{epoch}_it_{it}.jpg')
                plt.clf()
                sio.savemat(f'{vis_dir}/Sim_Phase.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})

            total_it += 1

        scheduler.step()

    t1 = time.time()
    print(f'Training takes {t1 - t0} seconds.')

    ############
    # Export final results
    out_errs = []
    for t in range(args.num_t):
        cur_t = (t / args.num_t) - 0.5
        cur_t = torch.FloatTensor([cur_t]).to(DEVICE)

        I_est, sim_g, sim_phs = net.get_estimates(cur_t)

        est_g = torch.exp(1j * sim_phs[0]).detach().cpu().squeeze().numpy()
        out_errs.append(np.uint8(ang_to_unit(np.angle(est_g)) * 255))

        if args.save_per_frame:
          sio.savemat(f'{vis_dir}/final/per_frame/sim_phase_{t}.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})

    I_est = np.uint8(torch.clamp(I_est, 0, 1).squeeze().detach().cpu().numpy() * 255)
    imageio.mimsave(f'{vis_dir}/final/final_turbulence.gif', out_errs, fps=30)
    imageio.imsave(f'{vis_dir}/final/final_I_est.png', I_est)

    print("Training concludes.")