import os, time, math, imageio, tqdm, argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from PIL import Image


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
# Hardcoded Zernike coefficients (start and end points)
delta = np.array([ 1.1416363 , -0.43582536, -0.06260428,  0.38009766,  0.25361097,
        -0.3148466 ,  0.92747134,  1.68069706, -2.32103207,  2.64558499,
        -1.35933578,  2.26350006,  0.47954541,  1.86205627,  0.23382413,
        -2.45344533, -0.90342144, -1.18094509, -2.21078039, -0.80214435,
        2.43905707,  0.38972859, -0.14482323,  0.57469502, -0.37923575,
        0.67929707, -0.37997045,  1.7486404 ])
alpha_0 = 2 * torch.FloatTensor([[-0.65455], [-2.34884], [-0.14921], [-1.76161], [-2.18389], [-3.22574], [-3.12588], [-0.04051],
[-2.15437], [-3.27762], [-3.45384], [-2.37893], [-2.00064], [-3.45384], [-2.37893], [-0.62289], [-2.18389], [-3.22574], [-3.12588], [-0.04051], 
[-3.45384], [-2.37893], [-0.62289], [-2.00064], [-0.62289], [-2.00064], [-2.08325], [-1.76161]]).to(DEVICE)
alpha_1 = alpha_0 - torch.FloatTensor(delta).unsqueeze(1).to(DEVICE) * 1.2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--img_name', default='Su_27_binary', type=str)
    parser.add_argument('--num_t', help='Number of patterns', default=256, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--width', default=128, type=int)
    parser.add_argument('--vis_freq', default=500, type=int)
    parser.add_argument('--init_lr', default=1e-3, type=float)
    parser.add_argument('--final_lr', default=1e-6, type=float)
    parser.add_argument('--silence_tqdm', action='store_true')
    parser.add_argument('--save_per_Frame', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)

    args = parser.parse_args()
    img_name = args.img_name
    PSF_size = args.width

    ############
    # Set up export directories
    data_dir = f'{args.root_dir}/data'
    vis_dir = f'{args.root_dir}/vis/{img_name}/res{args.width}_N{args.num_t}'
    os.makedirs(f'{args.root_dir}/vis', exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(f'{vis_dir}/final', exist_ok=True)
    if args.save_per_Frame:
      os.makedirs(f'{vis_dir}/final/per_frame', exist_ok=True)

    ############
    # Load in data
    I_obj = np.asarray(Image.open(f'{data_dir}/{img_name}.png').convert('L').resize((args.width, args.width))) / 255.
    DC = torchvision.transforms.Resize([args.width, args.width])(torch.tensor(I_obj).unsqueeze(0)).float().to(DEVICE)
    z_basis = compute_zernike_basis(num_polynomials=28, field_res=(PSF_size, PSF_size)).permute(1, 2, 0).to(DEVICE)

    ############
    # Simulate turbulence
    coeffs = []
    out_frames = []
    true_gs = []
    for i, t in enumerate(np.linspace(start=0, stop=1, num=args.num_t, endpoint=True)):
      coeff = (z_basis @ (t * alpha_1 + (1 - t) * alpha_0)).squeeze(2)
      g = torch.exp(1j * coeff)
      true_gs.append(g)
      coeffs.append(coeff)
      ave_coeffs = torch.mean(torch.stack(coeffs, 0))
      out_frames.append(np.uint8(255 * ang_to_unit(np.angle(g.cpu().detach().numpy()))))
    imageio.mimsave(f'{vis_dir}/turbulence.gif', out_frames, fps=60)

    ############
    # Simulate blurry measurements
    uncor_ys = []
    out_grayscale = []
    for i in range(args.num_t):
      _kernel = (fftshift(fft2(true_gs[i], norm="forward"), dim=[-2, -1]).abs() ** 2).unsqueeze(0)
      _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
      _y = F.conv2d(DC.unsqueeze(0), _kernel.unsqueeze(0), padding='same')
      uncor_ys.append(_y)
      out_grayscale.append(np.uint8(255 * _y.squeeze().cpu().numpy()))

      if i == 0 or i == args.num_t - 1:
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
        ax[3].imshow(DC.detach().cpu().squeeze(), cmap='gray')
        ax[3].axis('off')
        ax[3].title.set_text('True I')
        plt.savefig(f'{vis_dir}/Initialization_t{i}.jpg')
    imageio.mimsave(f'{vis_dir}/init_temporal.gif', out_grayscale, fps=60)

    ############
    # Simulate optical correction by conjugate of average turbulence
    corByAve_ys = []
    for i in range(args.num_t):
      _kernel = (fftshift(fft2( torch.exp(1j * (coeffs[i] - ave_coeffs)), norm="forward"), dim=[-2, -1]).abs() ** 2).unsqueeze(0)
      _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
      corByAve_ys.append(np.uint8(255 * F.conv2d(DC.unsqueeze(0), _kernel.unsqueeze(0), padding='same').squeeze().cpu().numpy()))
    imageio.mimsave(f'{vis_dir}/uncorByAve.gif', corByAve_ys, fps=60)

    ############
    # Initialize network
    net = TemporalZernNet(width=args.width, PSF_size=PSF_size)
    net = net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr)
    # Two cycles in CosineAnnealing scheduler
    # Not required
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.num_epochs // 2, eta_min=args.final_lr)

    ############
    # Load data
    xs, ys = preprocess_data(DC, true_gs)
    dset = RandomDataset(xs, ys)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    total_it = 0
    t = tqdm.trange(args.num_epochs, disable=args.silence_tqdm)

    ############
    # Training loop
    for epoch in t:
        for it, (x_batch, y_batch, idx) in enumerate(train_loader):
            x_batch, y_batch, idx = x_batch.to(DEVICE), y_batch.to(DEVICE), idx.to(DEVICE)
            optimizer.zero_grad()
            y, _kernel, sim_g, sim_phs, I_est = net(x_batch, idx / args.num_t - 0.5) # t from [-0.5, 0.5]

            mse_loss = F.mse_loss(y, y_batch)
            loss = mse_loss
            loss.backward()

            psnr = 10 * -torch.log10(mse_loss).item()
            t.set_postfix(MSE = f'{mse_loss.item():.4e}', PSNR = f'{psnr:.2f}')

            # Visualize results
            if args.vis_freq > 0 and ( (total_it + 1) % args.vis_freq) == 0:
                kkernel = fftshift(fft2(sim_g[0], norm="forward"), dim=[-2, -1]).abs() ** 2
                kkernel = kkernel / torch.sum(kkernel, dim=[-2, -1], keepdim=True)
                sim_uncor_y = F.conv2d(I_est, kkernel.unsqueeze(0), padding='same')

                conju = torch.exp(1j * -sim_phs[0])
                kernel = fftshift(fft2(true_gs[0] * conju, norm="forward"), dim=[-2, -1]).abs() ** 2
                kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
                y_conju = F.conv2d(DC.unsqueeze(0), kernel.unsqueeze(0), padding='same').squeeze()

                fig, ax = plt.subplots(1, 6, figsize=(40, 8))
                ax[0].imshow(y_batch[0].detach().cpu().squeeze(), cmap='gray')
                ax[0].axis('off')
                ax[0].title.set_text('Real I')
                ax[1].imshow(y[0].detach().cpu().squeeze(), cmap='gray')
                ax[1].axis('off')
                ax[1].title.set_text('Sim I')
                ax[2].imshow(I_est.detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
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

    ############
    # Export final results
    I_est = net.g_im()
    I_est = np.uint8(torch.clamp(I_est, 0, 1).squeeze().detach().cpu().numpy() * 255)
    out_corrected_frames = []
    out_est_turbulence = []
    for t in range(args.num_t):
        t_tensor = torch.ones(1, 1, 1, 1).to(DEVICE) * (t / args.num_t) - 0.5
        g_out = net.g_g(torch.cat([net.basis.repeat(len(t_tensor), 1, 1, 1).to(DEVICE), net.t_grid * t_tensor], dim = -1))
        sim_phs = g_out.permute(0, 3, 1, 2)
        conju = torch.exp(1j * -sim_phs[0])
        kernel = fftshift(fft2(true_gs[t] * conju, norm="forward"), dim=[-2, -1]).abs() ** 2
        kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
        y_conju_corrected = torch.clamp(F.conv2d(DC.unsqueeze(0), kernel.unsqueeze(0), padding='same').squeeze(), 0, 1)
        est_g = torch.exp(1j * sim_phs[0]).detach().cpu().squeeze().numpy()

        out_corrected_frames.append(np.uint8(y_conju_corrected.squeeze().detach().cpu().numpy() * 255))
        out_est_turbulence.append(np.uint8(ang_to_unit(np.angle(est_g)) * 255))
        if args.save_per_Frame:
          plt.imsave(f'{vis_dir}/final/per_frame/{t}.jpg', y_conju_corrected.squeeze().detach().cpu(), cmap='gray')
          sio.savemat(f'{vis_dir}/final/per_frame/sim_phase_{t}.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})

    imageio.mimsave(f'{vis_dir}/final/final_turbulence.gif', out_est_turbulence, fps=60)
    imageio.mimsave(f'{vis_dir}/final/final_corrected_frames.gif', out_corrected_frames, fps=60)
    imageio.imsave(f'{vis_dir}/final/final_I_est.png', I_est)

    ############
    # Compare with correcting the first frame through conjugate of [average turbulence] or [final turbulence]
    _kernel = (fftshift(fft2(torch.exp(1j * (coeffs[0] - coeffs[-1])), norm="forward"), dim=[-2, -1]).abs() ** 2).unsqueeze(0)
    _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
    fig, ax = plt.subplots(1, 4, figsize=(15, 8))
    ax[0].imshow(uncor_ys[0].squeeze().detach().cpu().squeeze(), cmap='gray')
    ax[0].axis('off')
    ax[0].title.set_text(f'Uncorrected at T=0')
    ax[1].imshow(F.conv2d(DC.unsqueeze(0), _kernel.unsqueeze(0), padding='same').squeeze().cpu().numpy(), cmap='gray')
    ax[1].axis('off')
    ax[1].title.set_text(f'Corrected by -g at T={args.num_t - 1}')
    ax[2].imshow(corByAve_ys[0], cmap='gray')
    ax[2].axis('off')
    ax[2].title.set_text(f'Corrected by average turbulence')
    ax[3].imshow(out_corrected_frames[0], cmap='gray')
    ax[3].axis('off')
    ax[3].title.set_text(f'Corrected by estimated at T=0')
    plt.savefig(f'{vis_dir}/final/final_comparison.jpg')
    plt.clf()

    print("Training Concludes.")