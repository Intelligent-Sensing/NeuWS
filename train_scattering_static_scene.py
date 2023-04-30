# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import os, time, imageio, tqdm, argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from PIL import Image

import torch
print(f"Using PyTorch Version: {torch.__version__}")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn.functional as F
import torchvision.transforms
from torch.fft import fft2, fftshift

from networks import TemporalZernNet, Rand_Abe_DIPNet

ang_to_unit = lambda x : ((x / np.pi) + 1) / 2

def preprocess_data(DC, true_gs):
    num = len(true_gs)
    x_list = []
    print('Preprocessing x_batch')
    for idx in range(num):
      p_SLM_train = 2 * np.pi * torch.rand(1, DC.shape[-2], DC.shape[-1]).float()  
      if idx == 0:
        p_SLM_train = torch.zeros_like(p_SLM_train)
      x_train = torch.exp(1j * p_SLM_train)
      x_list.append(x_train)
    y_list = []
    print('Preprocessing y_batch')
    for idx in range(num):
        kernel = fftshift(fft2(true_gs[idx].unsqueeze(0) * x_list[idx].to(DC.device), norm="forward"), dim=[-2, -1]).abs() ** 2
        kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
        kernel = kernel.unsqueeze(0)
        y = F.conv2d(DC.unsqueeze(0), kernel, padding='same').squeeze().cpu()
        y_list.append(y)
    return x_list, y_list

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, x_list, y_list):
        self.xs, self.ys = x_list, y_list

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], idx

DEVICE = 'cuda'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--img_name', default='Su_27_binary', type=str)
    parser.add_argument('--num_t', help='Number of patterns', default=256, type=int)
    parser.add_argument('--num_epochs', default=2000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--width', default=128, type=int)
    parser.add_argument('--vis_freq', default=20, type=int)
    parser.add_argument('--init_lr', default=1e-3, type=float)
    parser.add_argument('--final_lr', default=1e-3, type=float)
    parser.add_argument('--silence_tqdm', action='store_true')
    parser.add_argument('--save_per_frame', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dyn_abe', action='store_true')
    parser.add_argument('--zern_net', action='store_true')
    parser.add_argument('--circ_gaussian', action='store_true')
    parser.add_argument('--FFT', action='store_true')

    args = parser.parse_args()
    img_name = args.img_name
    PSF_size = args.width
    ############
    # Set up export directories
    data_dir = f'{args.root_dir}/data'
    vis_dir = f'{args.root_dir}/vis/new_scattering/{img_name}/{args.exp_name}_res{args.width}_N{args.num_t}'
    if args.dyn_abe:
      vis_dir += '_DynAbe'
    os.makedirs(f'{args.root_dir}/vis/new_scattering', exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(f'{vis_dir}/final', exist_ok=True)
    if args.save_per_frame:
      os.makedirs(f'{vis_dir}/final/per_frame', exist_ok=True)
    print(f'Saving output at: {vis_dir}')

    ############
    # Load in data
    I_obj = np.asarray(Image.open(f'{data_dir}/{img_name}.png').convert('L').resize((args.width, args.width))) / 255.
    DC = torchvision.transforms.Resize([args.width, args.width])(torch.tensor(I_obj).unsqueeze(0)).float().to(DEVICE)

    ############
    # Simulate turbulence
    out_frames = []
    true_gs = []

    if args.circ_gaussian:
      abe_0 = (1 / np.sqrt(2)) * torch.complex(
              torch.randn(PSF_size, PSF_size).float(),
              torch.randn(PSF_size, PSF_size).float())
      abe_1 = abe_0
      abe_0 = abe_0.to(DEVICE); abe_1 = abe_1.to(DEVICE)

      for i, t in enumerate(np.linspace(start=0, stop=1, num=args.num_t, endpoint=True)):
        g = abe_0
        true_gs.append(g)
        out_frames.append(np.uint8(255 * ang_to_unit(np.angle(g.cpu().detach().numpy()))))

    else:
      abe_0 = np.pi * (2 * torch.rand(PSF_size, PSF_size) - 1)
      if args.dyn_abe:
        abe_1 = abe_0 + 0.5 * torch.rand(PSF_size, PSF_size)
      else:
        abe_1 = abe_0
      abe_0 = abe_0.to(DEVICE); abe_1 = abe_1.to(DEVICE)

      coeffs = []
      for i, t in enumerate(np.linspace(start=0, stop=1, num=args.num_t, endpoint=True)):
        coeff = t * abe_0 + (1 - t) * abe_1
        g = torch.exp(1j * coeff)
        true_gs.append(g)
        coeffs.append(coeff)
        ave_coeffs = torch.mean(torch.stack(coeffs, 0))
        out_frames.append(np.uint8(255 * ang_to_unit(np.angle(g.cpu().detach().numpy()))))

    imageio.mimsave(f'{vis_dir}/turbulence.gif', out_frames[:120], fps=60)

    ############
    # Simulate blurry measurements
    uncor_ys = []
    out_grayscale = []
    for i in range(args.num_t):
      _kernel = (fftshift(fft2(true_gs[i], norm="forward"), dim=[-2, -1]).abs() ** 2).unsqueeze(0)
      _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
      _kernel = _kernel.unsqueeze(0).flip(2).flip(3)

      #_y = fft_2xPad_Conv2D(DC.unsqueeze(0), _kernel)
      _y = F.conv2d(DC.unsqueeze(0), _kernel, padding='same')
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
    imageio.mimsave(f'{vis_dir}/observations.gif', out_grayscale[:120], fps=60)

    ############
    # Simulate optical correction by conjugate of average turbulence
    corByAve_ys = []
    if not args.circ_gaussian:
      for i in range(args.num_t):
        _kernel = (fftshift(fft2( torch.exp(1j * (coeffs[i] - ave_coeffs)), norm="forward"), dim=[-2, -1]).abs() ** 2).unsqueeze(0)
        _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
        _kernel = _kernel.unsqueeze(0).flip(2).flip(3)
        corByAve_ys.append(np.uint8(255 * F.conv2d(DC.unsqueeze(0), _kernel, padding='same').squeeze().cpu().numpy()))

      imageio.mimsave(f'{vis_dir}/uncorByAve.gif', corByAve_ys[:120], fps=60)

    ############
    # Initialize network
    if args.zern_net:
      net = TemporalZernNet(width=args.width, PSF_size=PSF_size, use_FFT=args.FFT, phs_layers=8)
    else:
      net = Rand_Abe_DIPNet(width=args.width, PSF_size=PSF_size, use_FFT=args.FFT)
      #net = StaticDiffuseNet(width=args.width, PSF_size=PSF_size, phs_layers=8)

    net = net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr)
    # Two cycles in CosineAnnealing scheduler
    # Not required
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.num_epochs // 2, eta_min=args.final_lr)

    ############
    # Load data
    xs, ys = preprocess_data(DC, true_gs)
    dset = RandomDataset(xs, ys)

    x_batches = torch.cat(dset.xs, axis=0).unsqueeze(1).to(DEVICE)
    y_batches = torch.stack(dset.ys, axis=0).to(DEVICE)

    train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    total_it = 0
    t = tqdm.trange(args.num_epochs, disable=args.silence_tqdm)

    args.vis_freq = args.vis_freq * (len(dset) // args.batch_size)
    print(f'Visualize every {args.vis_freq} iters.')

    ############
    # Training loop
    t0 = time.time()
    for epoch in t:
        idxs=torch.randperm(len(dset)).long().to(DEVICE)
        for it in range(0,len(dset),args.batch_size):
            idx = idxs[it:it+args.batch_size]
            x_batch, y_batch = x_batches[idx], y_batches[idx]

            optimizer.zero_grad()
            y, _kernel, sim_g, sim_phs, I_est = net(x_batch, idx / args.num_t - 0.5) # t from [-0.5, 0.5]

            mse_loss = F.mse_loss(y, y_batch)
            loss = mse_loss
            loss.backward()

            #psnr = 10 * -torch.log10(mse_loss).item()
            t.set_postfix(MSE = f'{mse_loss.item():.4e}')#, PSNR = f'{psnr:.2f}')

            # Visualize results
            if args.vis_freq > 0 and ( (total_it + 1) % args.vis_freq) == 0:
                kkernel = fftshift(fft2(sim_g[0], norm="forward"), dim=[-2, -1]).abs() ** 2
                kkernel = kkernel / torch.sum(kkernel, dim=[-2, -1], keepdim=True)
                kkernel = kkernel.unsqueeze(0).flip(2).flip(3)
                sim_uncor_y = F.conv2d(I_est, kkernel, padding='same')

                conju = torch.exp(1j * -sim_phs[0])
                kernel = fftshift(fft2(true_gs[0] * conju, norm="forward"), dim=[-2, -1]).abs() ** 2
                kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
                kernel = kernel.unsqueeze(0).flip(2).flip(3)
                y_conju = F.conv2d(DC.unsqueeze(0), kernel, padding='same').squeeze()

                fig, ax = plt.subplots(1, 6, figsize=(40, 8))
                ax[0].imshow(y_batch[0].detach().cpu().squeeze(), cmap='gray')
                ax[0].axis('off')
                ax[0].title.set_text('Real I')
                ax[1].imshow(y[0].detach().cpu().squeeze(), cmap='gray')
                ax[1].axis('off')
                ax[1].title.set_text('Sim I')
                ax[2].imshow(torch.clamp(I_est, 0, 1).detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
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
    t1 = time.time()
    print(f'Training takes {t1 - t0} seconds.')

    ############
    # Export final results
    _, _, _, sim_phs, I_est = net(x_batch, torch.zeros_like(idx[:1]))
    I_est = np.uint8(torch.clamp(I_est, 0, 1).squeeze().detach().cpu().numpy() * 255)
    conju = torch.exp(1j * -sim_phs[0])
    kernel = fftshift(fft2(true_gs[0] * conju, norm="forward"), dim=[-2, -1]).abs() ** 2
    kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
    kernel = kernel.unsqueeze(0).flip(2).flip(3)
    y_conju = F.conv2d(DC.unsqueeze(0), kernel, padding='same').squeeze()
    y_conju = y_conju.squeeze().detach().cpu()
    y_conju = np.uint8(torch.clamp(y_conju, 0, 1).numpy() * 255)

    imageio.imsave(f'{vis_dir}/final/final_I_est.png', I_est)
    imageio.imsave(f'{vis_dir}/final/final_y_conju.png', y_conju)

    print("Training Concludes.")