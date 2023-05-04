# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import os, time, imageio, tqdm, argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

import torch
print(f"Using PyTorch Version: {torch.__version__}")
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

import torch.nn.functional as F
from torch.fft import fft2, fftshift
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
    parser.add_argument('--num_t', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--init_lr', default=1e-3, type=float)
    parser.add_argument('--final_lr', default=1e-3, type=float)
    parser.add_argument('--silence_tqdm', action='store_true')
    parser.add_argument('--save_per_frame', action='store_true')
    parser.add_argument('--static_phase', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_intensity', default=0, type=float)
    parser.add_argument('--im_prefix', default='SLM_raw', type=str)
    parser.add_argument('--zero_freq', default=-1, type=int)
    parser.add_argument('--phs_layers', default=2, type=int)
    parser.add_argument('--dynamic_scene', action='store_true')

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
    dset = BatchDataset(data_dir, num=args.num_t, im_prefix=args.im_prefix, max_intensity=args.max_intensity, zero_freq=args.zero_freq)
    x_batches = torch.cat(dset.xs, axis=0).unsqueeze(1).to(DEVICE)
    y_batches = torch.stack(dset.ys, axis=0).to(DEVICE)

    if args.dynamic_scene:
        net = MovingDiffuse(width=args.width, PSF_size=PSF_size, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase)
    else:
        net = StaticDiffuseNet(width=args.width, PSF_size=PSF_size, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase)

    net = net.to(DEVICE)

    im_opt = torch.optim.Adam(net.g_im.parameters(), lr=args.init_lr)
    ph_opt = torch.optim.Adam(net.g_g.parameters(), lr=args.init_lr)
    im_sche = torch.optim.lr_scheduler.CosineAnnealingLR(im_opt, T_max = args.num_epochs, eta_min=args.final_lr)
    ph_sche = torch.optim.lr_scheduler.CosineAnnealingLR(ph_opt, T_max = args.num_epochs, eta_min=args.final_lr)

    total_it = 0
    t = tqdm.trange(args.num_epochs, disable=args.silence_tqdm)

    ############
    # Training loop
    t0 = time.time()
    for epoch in t:
        idxs = torch.randperm(len(dset)).long().to(DEVICE)
        for it in range(0, len(dset), args.batch_size):
            idx = idxs[it:it+args.batch_size]
            x_batch, y_batch = x_batches[idx], y_batches[idx]
            cur_t = (idx / (args.num_t - 1)) - 0.5
            im_opt.zero_grad();  ph_opt.zero_grad()

            y, _kernel, sim_g, sim_phs, I_est = net(x_batch, cur_t)

            mse_loss = F.mse_loss(y, y_batch)

            loss = mse_loss
            loss.backward()

            ph_opt.step()
            im_opt.step()

            t.set_postfix(MSE=f'{mse_loss.item():.4e}')

            if args.vis_freq > 0 and (total_it % args.vis_freq) == 0:
                y, _kernel, sim_g, sim_phs, I_est = net(x_batch, torch.zeros_like(cur_t) - 0.5)

                Abe_est = fftshift(fft2(dset.a_slm.to(DEVICE) * sim_g, norm="forward"), dim=[-2, -1]).abs() ** 2
                if I_est.shape[0] > 1:
                    I_est = I_est[0:1]
                I_est = torch.clamp(I_est, 0, 1)
                yy = F.conv2d(I_est, Abe_est, padding='same').squeeze(0)

                fig, ax = plt.subplots(1, 6, figsize=(48, 8))
                ax[0].imshow(y_batch[0].detach().cpu().squeeze(), vmin=0, vmax=1, cmap='gray')
                ax[0].axis('off')
                ax[0].title.set_text('Real Measurement')
                ax[1].imshow(y[0].detach().cpu().squeeze(), vmin=0, vmax=1, cmap='gray')
                ax[1].axis('off')
                ax[1].title.set_text('Sim Measurement')
                ax[2].imshow(I_est.detach().cpu().squeeze(), cmap='gray')
                ax[2].axis('off')
                ax[2].title.set_text('I_est')
                ax[3].imshow(sim_phs[0].detach().cpu().squeeze() % np.pi, cmap='rainbow')
                ax[3].axis('off')
                ax[3].title.set_text(f'Sim Phase Error at t={idx[0]}')
                ax[4].imshow(_kernel[0].detach().cpu().squeeze(), cmap='gray')
                ax[4].axis('off')
                ax[4].title.set_text('Sim post-SLM PSF')
                ax[5].imshow(yy[0].squeeze().detach().cpu(), vmin=0, vmax=1, cmap='gray')
                ax[5].axis('off')
                ax[5].title.set_text(f'Abe_est * I_est at t={idx[0]}')
                plt.savefig(f'{vis_dir}/e_{epoch}_it_{it}.jpg')
                plt.clf()
                sio.savemat(f'{vis_dir}/Sim_Phase.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})

            total_it += 1

        im_sche.step()
        ph_sche.step()

    t1 = time.time()
    print(f'Training takes {t1 - t0} seconds.')

    ############
    # Export final results
    out_errs = []
    out_abes = []
    out_Iest = []
    for t in range(args.num_t):
        cur_t = (t / (args.num_t - 1)) - 0.5
        cur_t = torch.FloatTensor([cur_t]).to(DEVICE)

        I_est, sim_g, sim_phs = net.get_estimates(cur_t)
        I_est = torch.clamp(I_est, 0, 1).squeeze().detach().cpu().numpy()

        out_Iest.append(I_est)

        est_g = sim_g.detach().cpu().squeeze().numpy()
        out_errs.append(np.uint8(ang_to_unit(np.angle(est_g)) * 255))
        abe = sim_phs[0].detach().cpu().squeeze()
        abe = (abe - abe.min()) / (abe.max() - abe.min())
        out_abes.append(np.uint8(abe * 255))
        if args.save_per_frame and not args.static_phase:
          sio.savemat(f'{vis_dir}/final/per_frame/sim_phase_{t}.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})

    if args.dynamic_scene:
        out_Iest = [np.uint8(im * 255) for im in out_Iest]
        imageio.mimsave(f'{vis_dir}/final/final_I.gif', out_Iest, duration=1000*1./30)
    else:
        I_est = np.uint8(I_est.squeeze() * 255)
        imageio.imsave(f'{vis_dir}/final/final_I_est.png', I_est)

    if args.static_phase:
        imageio.imsave(f'{vis_dir}/final/final_aberrations_angle.png', out_errs[0])
        imageio.imsave(f'{vis_dir}/final/final_aberrations.png', out_abes[0])
    else:
        imageio.mimsave(f'{vis_dir}/final/final_aberrations_angle_grey.gif', out_errs, duration=1000*1./30)
        imageio.mimsave(f'{vis_dir}/final/final_aberrations.gif', out_abes, duration=1000*1./30)

    print("Training concludes.")

    colored_err = []
    for i, a in enumerate(out_errs):
        plt.imsave(f'{vis_dir}/final/per_frame/{i:03d}.jpg', a, cmap='rainbow')
        colored_err.append(imageio.imread(f'{vis_dir}/final/per_frame/{i:03d}.jpg'))
    imageio.mimsave(f'{vis_dir}/final/final_aberrations_angle.gif', colored_err, duration=1000*1./30)