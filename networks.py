# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.fft import fft2, fftshift

import numpy as np
import torchvision.transforms
from utils import compute_zernike_basis, fft_2xPad_Conv2D


class sine_act(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.sin(x)
        return out


class G_Renderer(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=32, num_layers=2, out_dim=1):
        super().__init__()
        act_fn = nn.ReLU()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            #layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class G_FeatureTensor(nn.Module):
    def __init__(self, x_dim, y_dim, num_feats = 32, ds_factor = 1):
        super().__init__()
        self.x_dim, self.y_dim = x_dim, y_dim
        x_mode, y_mode = x_dim // ds_factor, y_dim // ds_factor
        self.num_feats = num_feats

        self.data = nn.Parameter(
            0.1 * torch.randn((x_mode, y_mode, num_feats)), requires_grad=True)

        half_dx, half_dy =  0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1-half_dx, x_dim)
        ys = torch.linspace(half_dx, 1-half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()

        xs = xy * torch.tensor([x_mode, y_mode], device=xs.device).float()
        indices = xs.long()
        self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

        self.x0 = nn.Parameter(indices[:, 0].clamp(min=0, max=x_mode-1), requires_grad=False)
        self.y0 = nn.Parameter(indices[:, 1].clamp(min=0, max=y_mode-1), requires_grad=False)
        self.x1 = nn.Parameter((self.x0 + 1).clamp(max=x_mode-1), requires_grad=False)
        self.y1 = nn.Parameter((self.y0 + 1).clamp(max=y_mode-1), requires_grad=False)

    def sample(self):
        return (
				self.data[self.y0, self.x0] * (1.0 - self.lerp_weights[:,0:1]) * (1.0 - self.lerp_weights[:,1:2]) +
				self.data[self.y0, self.x1] * self.lerp_weights[:,0:1] * (1.0 - self.lerp_weights[:,1:2]) +
				self.data[self.y1, self.x0] * (1.0 - self.lerp_weights[:,0:1]) * self.lerp_weights[:,1:2] +
				self.data[self.y1, self.x1] * self.lerp_weights[:,0:1] * self.lerp_weights[:,1:2]
			)

    def forward(self):
        return self.sample()


class G_Tensor(G_FeatureTensor):
    def __init__(self, x_dim, y_dim=None):
        if y_dim is None:
            y_dim = x_dim
        super().__init__(x_dim, y_dim)
        self.renderer = G_Renderer()

    def forward(self):
        feats = self.sample()
        return self.renderer(feats).reshape([-1, 1, self.x_dim, self.y_dim])


class G_PatchTensor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net1 = G_Tensor(width // 2, width // 2)
        self.net2 = G_Tensor(width // 2, width // 2)
        self.net3 = G_Tensor(width // 2, width // 2)
        self.net4 = G_Tensor(width // 2, width // 2)

    def forward(self):
        p1 = self.net1()
        p2 = self.net2()
        p3 = self.net3()
        p4 = self.net4()

        left = torch.cat([p1, p2], axis=-1)
        right = torch.cat([p3, p4], axis=-1)

        return torch.cat([left, right], axis=-2)


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class G_SpaceTime(nn.Module):
    def __init__(self, x_width, y_width, bsize=8):
        super().__init__()

        hidden_dim, num_hidden_layers = 32, 3

        self.spatial_net = G_FeatureTensor(x_width, y_width, hidden_dim, ds_factor=1)
        self.x_width, self.y_width = x_width, y_width

        self.t0 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.t1 = nn.Parameter(torch.randn(1), requires_grad=True)

        num_t_freq = 5
        self.embedding_t = Embedding(1, num_t_freq) 
        #t_dim = 1 + num_t_freq * 2 + 2
        t_dim = 1 + 2

        act_fn = sine_act()#nn.LeakyReLU(inplace=True)
        layers = []
        layers.append(nn.Linear(t_dim, hidden_dim))
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, 2))

        self.warp_net = nn.Sequential(*layers)

        xs = torch.linspace(-1, 1, steps=x_width)
        ys = torch.linspace(-1, 1, steps=y_width)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        self.xy_basis = nn.Parameter(
            torch.stack([x, y], axis=-1).unsqueeze(0).repeat(bsize, 1, 1, 1),
            requires_grad=False
        )
        self.renderer = G_Renderer(in_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=3)

    def forward(self, t):
        spatial_feats = self.spatial_net().unsqueeze(0).repeat(t.shape[0], 1, 1)
        spatial_feats = spatial_feats.reshape(-1, self.x_width, self.y_width, spatial_feats.shape[-1])
        spatial_feats = spatial_feats.permute(0, 3, 1, 2)

        alpha = (t + 0.5).unsqueeze(-1)
        t_emb = alpha * torch.ones_like(self.t1)
        #t_emb = (t).unsqueeze(-1) * torch.ones_like(self.t1)
        t_emb = t_emb.unsqueeze(-2).unsqueeze(-2).repeat(1, self.x_width, self.y_width, 1)
        #t_emb = self.embedding_t(t_emb)

        t_input = torch.cat([self.xy_basis[:t.shape[0]], t_emb], axis=-1)

        motion = self.warp_net(t_input) * 0.1

        feats = F.grid_sample(spatial_feats, motion + self.xy_basis[:t.shape[0]])
        output = self.renderer(feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        output = F.leaky_relu(output, 0.001)

        return output


class TemporalZernNet(nn.Module):
    def __init__(self, width, PSF_size, phs_layers = 2, use_FFT=True, bsize=8, use_pe=False, static_phase=True):
        super().__init__()
        self.g_im = G_PatchTensor(width)

        if not use_pe:
            self.basis = nn.Parameter(compute_zernike_basis(
                num_polynomials=28,
                field_res=(PSF_size, PSF_size)).permute(1, 2, 0).unsqueeze(0).repeat(bsize, 1, 1, 1),
                requires_grad=False)
        else:
            xs = torch.linspace(-1, 1, steps=PSF_size)
            ys = torch.linspace(-1, 1, steps=PSF_size)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            basis = []
            for i in range(1, 16):
                basis.append(torch.sin(i * x))
                basis.append(torch.sin(i * y))
                basis.append(torch.cos(i * x))
                basis.append(torch.cos(i * y))
            self.basis = nn.Parameter(
                torch.stack(basis, axis=-1).unsqueeze(0).repeat(bsize, 1, 1, 1),
                requires_grad=False
            )

        hidden_dim = 32
        t_dim = 1 if not static_phase else 0
        in_dim = self.basis.shape[-1]

        act_fn = nn.LeakyReLU(inplace=True)
        layers = []
        layers.append(nn.Linear(t_dim + in_dim, hidden_dim))
        for _ in range(phs_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_dim, 1))
        self.g_g = nn.Sequential(*layers)

        self.t_grid = nn.Parameter(torch.ones_like(self.basis[..., 0:1]), requires_grad=False)
        self.use_FFT = use_FFT
        print(f'Using FFT approximation of convolution: {self.use_FFT}')
        self.static_phase = static_phase

    def get_estimates(self, t):
        I_est = self.g_im()

        if not self.static_phase:
            cat_grid = self.t_grid[:t.shape[0]] * t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            g_in = torch.cat([self.basis[:t.shape[0]], cat_grid], dim = -1)

        else:
            g_in = self.basis[:t.shape[0]]

        g_out = self.g_g(g_in)
        sim_phs = g_out.permute(0, 3, 1, 2)
        sim_g = torch.exp(1j * sim_phs)

        return I_est, sim_g, sim_phs

    def forward(self, x_batch, t):
        I_est, sim_g, sim_phs = self.get_estimates(t)
        _kernel = fftshift(fft2(sim_g * x_batch, norm="forward"), dim=[-2, -1]).abs() ** 2
        _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
        _kernel = _kernel.flip(2).flip(3)

        if self.use_FFT:
            y = fft_2xPad_Conv2D(I_est, _kernel).squeeze()
        else:
            y = F.conv2d(I_est, _kernel, padding='same').squeeze()

        return y, _kernel, sim_g, sim_phs, I_est


class StaticDiffuseNet(TemporalZernNet):
    def __init__(self, width, PSF_size, phs_layers = 2, use_FFT=True, bsize=8, use_pe=False, static_phase=True):
        super().__init__(width, PSF_size, phs_layers = phs_layers, use_FFT=use_FFT, bsize=bsize, use_pe=use_pe)

        hidden_dim = 32
        t_dim = 1 if not static_phase else 0
        in_dim = self.basis.shape[-1]
        act_fn = nn.LeakyReLU(inplace=True)
        layers = []
        layers.append(nn.Linear(t_dim + in_dim, hidden_dim))
        for _ in range(phs_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            # layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, 2))
        self.g_g = nn.Sequential(*layers)
        self.static_phase = static_phase

    def get_estimates(self, t):
        I_est = self.g_im()

        if not self.static_phase:
            cat_grid = self.t_grid[:t.shape[0]] * t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            g_in = torch.cat([self.basis[:t.shape[0]], cat_grid], dim = -1)
        else:
            cat_grid = self.t_grid[:t.shape[0]]
            g_in = self.basis[:t.shape[0]]

        g_out = self.g_g(g_in)

        g_out = g_out.permute(0, 3, 1, 2)
        sim_phs = g_out[:, 1:2]
        sim_amp = g_out[:, 0:1]
        sim_g = sim_amp * torch.exp(1j * sim_phs)

        return I_est, sim_g, sim_phs

    # def forward(self, x_batch, t):
    #     I_est, sim_g, sim_phs = self.get_estimates(t)
    #     _kernel = fftshift(fft2(sim_g * x_batch, norm="forward"), dim=[-2, -1]).abs() ** 2
    #     _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
    #     _kernel = _kernel.flip(2).flip(3)
    #
    #     if self.use_FFT:
    #         y = fft_2xPad_Conv2D(I_est, _kernel).squeeze()
    #     else:
    #         y = F.conv2d(I_est, _kernel, padding='same').squeeze()
    #
    #     return y, _kernel, sim_g, sim_phs, I_est


class MovingTemporalZernNet(TemporalZernNet):
    def __init__(self, width, PSF_size, phs_layers = 5, use_FFT=True, bsize=8, use_pe=False, static_phase=True):
        super().__init__(width, PSF_size, use_pe=False, phs_layers=phs_layers, use_FFT=use_FFT, bsize=bsize)
        self.g_im = G_SpaceTime(width, width, bsize)

        in_dim = self.basis.shape[-1]

        t_dim = 3 if not static_phase else 0
        hidden_dim = 32
        act_fn = nn.LeakyReLU(inplace=True)
        layers = []
        layers.append(nn.Linear(in_dim + t_dim, hidden_dim))
        for _ in range(phs_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, 1))
        self.g_g = nn.Sequential(*layers)
        self.static_phase = static_phase

    def get_estimates(self, t):
        I_est = self.g_im(t)
        g_in = self.basis[:t.shape[0]]

        if not self.static_phase:
            cat_grid = self.t_grid[:t.shape[0]] * t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            cat_grid = torch.cat([cat_grid, torch.cos(cat_grid), torch.sin(cat_grid)], dim = -1)
            g_in = torch.cat([self.basis[:t.shape[0]], cat_grid], dim = -1)

        g_out = self.g_g(g_in)

        sim_phs = g_out.permute(0, 3, 1, 2)
        # Unit circle constraint
        sim_g = torch.exp(1j * sim_phs)

        return I_est, sim_g, sim_phs

    def forward(self, x_batch, t):
        I_est, sim_g, sim_phs = self.get_estimates(t)

        _kernel = fftshift(fft2(sim_g * x_batch, norm="forward"), dim=[-2, -1]).abs() ** 2
        _factor = torch.sum(_kernel, dim=[-2, -1], keepdim=True)
        _kernel = _kernel / _factor
        _kernel = _kernel.flip(2).flip(3)

        if self.use_FFT:
            y = []
            for i in range(len(t)):
                y.append(fft_2xPad_Conv2D(I_est[i:i+1], _kernel[i:i+1]).squeeze())
            y = torch.stack(y, axis=0)
            #y = fft_2xPad_Conv2D(I_est.permute(1, 0, 2, 3), _kernel, groups=4).squeeze()
        else:
            y = []
            for i in range(len(t)):
                y.append(F.conv2d(I_est[i:i+1], _kernel[i:i+1], padding='same').squeeze())
            y = torch.stack(y, axis=0)

        return y, _kernel, sim_g, sim_phs, I_est


class MovingDiffuse(TemporalZernNet):
    def __init__(self, width, PSF_size, phs_layers = 5, use_FFT=True, bsize=8, use_pe=False, static_phase=True):
        super().__init__(width, PSF_size, phs_layers=phs_layers, use_FFT=use_FFT, bsize=bsize, static_phase=static_phase)
        self.g_im = G_SpaceTime(width, width, bsize)

        self.PSF_size = PSF_size

        t_dim = 3 if not static_phase else 0
        in_dim = self.basis.shape[-1]
        hidden_dim = 32

        act_fn = nn.LeakyReLU(inplace=True)
        layers = []
        layers.append(nn.Linear(t_dim + in_dim, hidden_dim))
        for _ in range(phs_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, 2))

        self.g_g = nn.Sequential(*layers)
        self.static_phase = static_phase

    def get_estimates(self, t, return_amp=False):

        I_est = self.g_im(t)

        if not self.static_phase:
            cat_grid = self.t_grid[:t.shape[0]] * t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            cat_grid = torch.cat([cat_grid, torch.cos(cat_grid), torch.sin(cat_grid)], dim = -1)
            g_in = torch.cat([self.basis[:t.shape[0]], cat_grid], dim = -1)

        else:
            g_in = self.basis[:t.shape[0]]

        g_out = self.g_g(g_in)
        g_out = g_out.permute(0, 3, 1, 2)

        sim_phs = g_out[:, 1:2]
        sim_amp = g_out[:, 0:1]
        sim_g = sim_amp * torch.exp(1j * sim_phs)
        if return_amp:
            return I_est, sim_g, sim_phs, sim_amp

        return I_est, sim_g, sim_phs,

    def forward(self, x_batch, t):
        I_est, sim_g, sim_phs = self.get_estimates(t)

        _kernel = fftshift(fft2(sim_g * x_batch, norm="forward"), dim=[-2, -1]).abs() ** 2
        _factor = torch.sum(_kernel, dim=[-2, -1], keepdim=True)
        _kernel = _kernel / _factor
        _kernel = _kernel.flip(2).flip(3)

        if self.use_FFT:
            y = []
            for i in range(len(t)):
                y.append(fft_2xPad_Conv2D(I_est[i:i+1], _kernel[i:i+1]).squeeze())
            y = torch.stack(y, axis=0)
        else:
            y = []
            for i in range(len(t)):
                y.append(F.conv2d(I_est[i:i+1], _kernel[i:i+1], padding='same').squeeze())
            y = torch.stack(y, axis=0)

        return y, _kernel, sim_g, sim_phs, I_est




