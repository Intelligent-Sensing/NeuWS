# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import os, time, math, imageio, tqdm, argparse
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import numpy as np
from PIL import Image

import torch
print(torch.__version__)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torch.fft import fft2, ifft2, fftshift
from networks import *
from utils import *
from dataset import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--img_name', default='moving_digits', type=str)
    parser.add_argument('--generate_data', action='store_true')
    args = parser.parse_args()

    device = 'cuda'

    data_dir = f'{args.root_dir}/data'
    img_name = args.img_name
    if args.generate_data:
        gen_moving_dataset(data_dir)

    num_t = 256
    vis_freq = 500
    num_epochs = 1000
    n_batch = 8
    width = 128
    PSF_size = width

    vis_dir = f'{args.root_dir}/vis/{img_name}/res{width}_N256'
    os.makedirs(f'{args.root_dir}/vis', exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    ang_to_unit = lambda x : ((x / np.pi) + 1) / 2

    DC = torchvision.transforms.Resize([width, width])(torch.FloatTensor((np.load(f'{data_dir}/moving_digits_n256.npy') / 255.))).to(device)

    rand_perturb = np.array([ 1.1416363 , -0.43582536, -0.06260428,  0.38009766,  0.25361097,
           -0.3148466 ,  0.92747134,  1.68069706, -2.32103207,  2.64558499,
           -1.35933578,  2.26350006,  0.47954541,  1.86205627,  0.23382413,
           -2.45344533, -0.90342144, -1.18094509, -2.21078039, -0.80214435,
            2.43905707,  0.38972859, -0.14482323,  0.57469502, -0.37923575,
            0.67929707, -0.37997045,  1.7486404 ])

    z_basis = compute_zernike_basis(num_polynomials=28, field_res=(PSF_size, PSF_size)).permute(1, 2, 0).to(device)
    alpha_0 = 2 * torch.FloatTensor([[-0.65455], [-2.34884], [-0.14921], [-1.76161], [-2.18389], [-3.22574], [-3.12588], [-0.04051],
    [-2.15437], [-3.27762], [-3.45384], [-2.37893], [-2.00064], [-3.45384], [-2.37893], [-0.62289], [-2.18389], [-3.22574], [-3.12588], [-0.04051], 
    [-3.45384], [-2.37893], [-0.62289], [-2.00064], [-0.62289], [-2.00064], [-2.08325], [-1.76161]]).to(device)
    alpha_1 = alpha_0 - torch.FloatTensor(rand_perturb).unsqueeze(1).to(device) * 1.2

    a_slm = np.ones((72, 128))
    a_slm = np.lib.pad(a_slm, (((128 - 72) // 2, (128 - 72) // 2), (0, 0)), 'constant', constant_values=(0, 0))

    coeffs = []
    for t in np.linspace(start=0, stop=1, num=num_t, endpoint=True):
      coeffs.append((z_basis @ (t * alpha_1 + (1 - t) * alpha_0)).squeeze(2))
    ave_coeffs = torch.mean(torch.stack(coeffs, 0))
    true_gs = []
    for i, t in enumerate(np.linspace(start=0, stop=1, num=num_t, endpoint=True)):
      true_gs.append(torch.exp(1j * coeffs[i]))

    out_frames = []
    for g in true_gs:
      out_frames.append(np.uint8(255 * ang_to_unit(np.angle(g.cpu().detach().numpy()))))
    imageio.mimsave(f'{vis_dir}/turbulence.gif', out_frames, fps=60)

    out = []
    for i in range(num_t):
      kernel = fftshift(fft2(true_gs[i].unsqueeze(0), norm="forward"), dim=[-2, -1]).abs() ** 2
      kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
      kernel = kernel.unsqueeze(0)
      out.append(np.uint8(255 * F.conv2d(DC[i:(i+1)].unsqueeze(0), kernel, padding='same').squeeze().cpu().numpy()))
    imageio.mimsave(f'{vis_dir}/init_temporal.gif', out, fps=60)

    uncor_ys = []
    for i in range(num_t):
      _kernel = (fftshift(fft2(true_gs[i], norm="forward"), dim=[-2, -1]).abs() ** 2).unsqueeze(0)
      _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
      uncor_ys.append(F.conv2d(DC[i:(i+1)].unsqueeze(0), _kernel.unsqueeze(0), padding='same'))
      if i == 0 or i == num_t - 1:
        fig, ax = plt.subplots(1, 4, figsize=(24, 8))
        ax[0].imshow(uncor_ys[i].squeeze().detach().cpu().squeeze(), cmap='gray')
        ax[0].axis('off')
        ax[0].title.set_text(f'Uncorrected I t{i}')
        ax[1].imshow(_kernel.detach().cpu().squeeze(), cmap='gray')
        ax[1].axis('off')
        ax[1].title.set_text(f'Uncorrected PSF t{i}')
        ax[2].imshow(ang_to_unit(np.angle(true_gs[i].cpu().detach().numpy())), cmap='gray')
        ax[2].axis('off')
        ax[2].title.set_text(f'Uncorrected Phase Error t{i}')
        ax[3].imshow(DC[i].detach().cpu().squeeze(), cmap='gray')
        ax[3].axis('off')
        ax[3].title.set_text('True I')
        plt.savefig(f'{vis_dir}/Initialization_t{i}.jpg')

    corByAve_ys = []
    for i in range(num_t):
      _kernel = (fftshift(fft2( torch.exp(1j * (coeffs[i] - ave_coeffs)), norm="forward"), dim=[-2, -1]).abs() ** 2).unsqueeze(0)
      _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
      corByAve_ys.append(np.uint8(255 * F.conv2d(DC[i:(i+1)].unsqueeze(0), _kernel.unsqueeze(0), padding='same').squeeze().cpu().numpy()))
    imageio.mimsave(f'{vis_dir}/uncorByAve.gif', corByAve_ys, fps=60)

    _kernel = (fftshift(fft2(torch.exp(1j * (coeffs[0] - coeffs[-1])), norm="forward"), dim=[-2, -1]).abs() ** 2).unsqueeze(0)
    _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(uncor_ys[0].squeeze().detach().cpu().squeeze(), cmap='gray')
    ax[0].axis('off')
    ax[0].title.set_text(f'Uncorrected at T=0')
    ax[1].imshow(F.conv2d(DC[0:1].unsqueeze(0), _kernel.unsqueeze(0), padding='same').squeeze().cpu().numpy(), cmap='gray')
    ax[1].axis('off')
    ax[1].title.set_text(f'Corrected by -g at T=255')
    ax[2].imshow(corByAve_ys[0], cmap='gray')
    ax[2].axis('off')
    ax[2].title.set_text(f'Corrected by average turbulence')

    net = TemporalZernNet(width=width, PSF_size=PSF_size)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs // 2, eta_min=1e-6)

    xs, ys = preprocess_data(DC, true_gs)
    dset = RandomDataset(xs, ys)

    train_loader = torch.utils.data.DataLoader(dset, batch_size=n_batch, num_workers=2, shuffle=True)
    total_it = 0
    t = tqdm.trange(num_epochs)
    for epoch in t:
        for it, (x_batch, y_batch, idx) in enumerate(train_loader):
            x_batch, y_batch, idx = x_batch.to(device), y_batch.to(device), idx.to(device)
            optimizer.zero_grad()
            y, _kernel, sim_g, sim_phs, I_est = net(x_batch, idx / num_t - 0.5)

            mse_loss = F.mse_loss(y, y_batch)
            loss = mse_loss
            loss.backward()

            psnr = 10 * -torch.log10(mse_loss).item()
            t.set_postfix(MSE = f'{mse_loss.item():.4e}', PSNR = f'{psnr:.2f}')

            if ( (total_it + 1) % vis_freq) == 0 or total_it == 0:
                kkernel = fftshift(fft2(sim_g[0], norm="forward"), dim=[-2, -1]).abs() ** 2
                kkernel = kkernel / torch.sum(kkernel, dim=[-2, -1], keepdim=True)
                sim_uncor_y = F.conv2d(I_est[0:1], kkernel.unsqueeze(0), padding='same')

                conju = torch.exp(1j * -sim_phs[0])
                kernel = fftshift(fft2(true_gs[idx[0]] * conju, norm="forward"), dim=[-2, -1]).abs() ** 2
                kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
                y_conju = F.conv2d(DC[idx[0]].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0), padding='same').squeeze()

                fig, ax = plt.subplots(1, 6, figsize=(40, 8))
                ax[0].imshow(y_batch[0].detach().cpu().squeeze(), cmap='gray')
                ax[0].axis('off')
                ax[0].title.set_text('Real I')
                ax[1].imshow(y[0].detach().cpu().squeeze(), cmap='gray')
                ax[1].axis('off')
                ax[1].title.set_text('Sim I')
                ax[2].imshow(I_est[0:1].detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                ax[2].axis('off')
                ax[2].title.set_text('I_est')
                ax[3].imshow(ang_to_unit(np.angle(sim_g[0].detach().cpu().squeeze().numpy())), cmap='gray')
                ax[3].axis('off')
                ax[3].title.set_text('Sim Phase')
                ax[4].imshow(sim_uncor_y.squeeze().detach().cpu(), cmap='gray')
                ax[4].axis('off')
                ax[4].title.set_text('Captured w/ Sim_g0')
                ax[5].imshow(y_conju.squeeze().detach().cpu(), cmap='gray')
                ax[5].axis('off')
                ax[5].title.set_text('Sim Conjugate')
                plt.savefig(f'{vis_dir}/e_{epoch}_it_{it}.jpg')
                plt.clf()
                sio.savemat(f'{vis_dir}/Sim_Phase.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})

            optimizer.step()
            total_it += 1
        scheduler.step()

    out_Iest = []
    out_frames = []
    out_errs = []
    os.makedirs(f'{vis_dir}/final_temporal', exist_ok=True)
    
    for t in range(num_t):
        t_tensor = torch.ones(1, 1, 1, 1).to(device) * (t / num_t) - 0.5
        g_out = net.g_g(torch.cat([net.basis.repeat(len(t_tensor), 1, 1, 1).to(device), net.t_grid * t_tensor], dim = -1))
        sim_phs = g_out.permute(0, 3, 1, 2)

        I_est = net.g_im(t_tensor)
        out_Iest.append(np.uint8(torch.clamp(I_est, 0, 1).squeeze().detach().cpu().numpy() * 255))

        conju = torch.exp(1j * -sim_phs[0])
        kernel = fftshift(fft2(true_gs[t] * conju, norm="forward"), dim=[-2, -1]).abs() ** 2
        kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
        y_conju = torch.clamp(F.conv2d(DC[t:(t+1)].unsqueeze(0), kernel.unsqueeze(0), padding='same').squeeze(), 0, 1)
        plt.imsave(f'{vis_dir}/final_temporal/{t}.jpg', y_conju.squeeze().detach().cpu(), cmap='gray')
        out_frames.append(np.uint8(y_conju.squeeze().detach().cpu().numpy() * 255))
        est_g = torch.exp(1j * sim_phs[0]).detach().cpu().squeeze().numpy()
        out_errs.append(np.uint8(ang_to_unit(np.angle(est_g)) * 255))

    imageio.mimsave(f'{vis_dir}/final_turbulence.gif', out_errs, fps=60)
    imageio.mimsave(f'{vis_dir}/final_temporal.gif', out_frames, fps=60)
    imageio.mimsave(f'{vis_dir}/final_Iest.gif', out_Iest, fps=60)

    _kernel = (fftshift(fft2(torch.exp(1j * (coeffs[0] - coeffs[-1])), norm="forward"), dim=[-2, -1]).abs() ** 2).unsqueeze(0)
    _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
    fig, ax = plt.subplots(1, 4, figsize=(15, 8))
    ax[0].imshow(uncor_ys[0].squeeze().detach().cpu().squeeze(), cmap='gray')
    ax[0].axis('off')
    ax[0].title.set_text(f'Uncorrected at T=0')
    ax[1].imshow(F.conv2d(DC[0:1].unsqueeze(0), _kernel.unsqueeze(0), padding='same').squeeze().cpu().numpy(), cmap='gray')
    ax[1].axis('off')
    ax[1].title.set_text(f'Corrected by -g at T={num_t - 1}')
    ax[2].imshow(corByAve_ys[0], cmap='gray')
    ax[2].axis('off')
    ax[2].title.set_text(f'Corrected by average turbulence')
    ax[3].imshow(out_frames[0], cmap='gray')
    ax[3].axis('off')
    ax[3].title.set_text(f'Corrected by estimated at T=0')
    plt.savefig(f'{vis_dir}/final_comparison.jpg')
    plt.clf()

    _kernel = (fftshift(fft2(torch.exp(1j * (coeffs[0] - coeffs[-1])), norm="forward"), dim=[-2, -1]).abs() ** 2).unsqueeze(0)
    _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
    fig, ax = plt.subplots(1, 4, figsize=(15, 8))
    ax[0].imshow(uncor_ys[0].squeeze().detach().cpu().squeeze(), cmap='gray')
    ax[0].axis('off')
    ax[0].title.set_text(f'Uncorrected at T=0')
    ax[1].imshow(F.conv2d(DC[0:1].unsqueeze(0), _kernel.unsqueeze(0), padding='same').squeeze().cpu().numpy(), cmap='gray')
    ax[1].axis('off')
    ax[1].title.set_text(f'Corrected by -g at T={num_t - 1}')
    ax[2].imshow(corByAve_ys[0], cmap='gray')
    ax[2].axis('off')
    ax[2].title.set_text(f'Corrected by average turbulence')
    ax[3].imshow(out_frames[0], cmap='gray')
    ax[3].axis('off')
    ax[3].title.set_text(f'Corrected by estimated at T=0')
    plt.savefig(f'{vis_dir}/final_comparison.jpg')
    plt.clf()
    
    print("Done")
