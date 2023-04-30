# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import os, tqdm
from aotools.functions import zernikeArray
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift, irfftn, rfftn
from networks import *

ang_to_unit = lambda x : ((x / np.pi) + 1) / 2

def gen_moving_dataset(data_dir):
    nums_per_image = 2
    original_size = 128
    width, height = 224, 224

    lims = (x_lim, y_lim) = width - original_size, height - original_size
    num_frames = 256

    #direcs = np.pi * (np.random.rand(nums_per_image) * 2 - 1)
    direcs = np.array([0.97302908, -0.96160664])
    #speeds = (np.random.randint(5, size=nums_per_image) + 2) / 10
    speeds = np.array([0.3, 0.3]) * (128 / num_frames)

    veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])
    # Get a list containing two PIL images randomly sampled from the database
    mnist_images = [Image.open(f'{data_dir}/digit_4.png').convert('L').resize((original_size, original_size),Image.ANTIALIAS), 
                    Image.open(f'{data_dir}/digit_0.png').convert('L').resize((original_size, original_size),Image.ANTIALIAS)]
    # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
    positions = np.asarray([[15, 15], [75, 75]])

    dataset = np.empty((num_frames, 128, 128), dtype=np.uint8)

    # Generate new frames for the entire num_framesgth
    for frame_idx in range(num_frames):

        canvases = [Image.new('L', (width, height)) for _ in range(nums_per_image)]
        canvas = np.zeros((1, width, height), dtype=np.float32)

        # In canv (i.e Image object) place the image at the respective positions
        # Super impose both images on the canvas (i.e empty np array)
        for i, canv in enumerate(canvases):
            canv.paste(mnist_images[i], tuple(positions[i].astype(int)))
            canvas += np.asarray(canv)

        # Get the next position by adding velocity
        next_pos = positions + veloc

        # Iterate over velocity and see if we hit the wall
        # If we do then change the  (change direction)
        for i, pos in enumerate(next_pos):
            for j, coord in enumerate(pos):
                if coord < -2 or coord > lims[j] + 2:
                    veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

        # Make the permanent change to position by adding updated velocity
        positions = positions + veloc
        frame_out = (canvas).clip(0, 255).astype(np.uint8)
        dataset[frame_idx] = np.asarray(Image.fromarray(frame_out.squeeze()).resize((128, 128), Image.LANCZOS))
    plt.imshow(dataset[0], cmap='gray')

    imageio.mimsave(f'{data_dir}/moving_digits_n256.gif', dataset, fps=60)
    np.save(f'{data_dir}/moving_digits_n256.npy', dataset)

def preprocess_sim_dynamic_data(DC, true_gs):
    num = len(true_gs)
    x_list = []
    print('Loading into RAM')
    for idx in tqdm.trange(num):
      p_SLM_train = 2 * np.pi * torch.rand(1, 128, 128).float()
      if idx == num // 2:
        p_SLM_train = torch.zeros_like(p_SLM_train)
      x_train = torch.exp(1j * p_SLM_train)
      x_list.append(x_train)
    y_list = []
    print('Preprocessing y_batch')
    for idx in tqdm.trange(num):
      kernel = fftshift(fft2(true_gs[idx].unsqueeze(0) * x_list[idx].to(DC.device), norm="forward"), dim=[-2, -1]).abs() ** 2
      kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
      kernel = kernel.unsqueeze(0)
      y_list.append(F.conv2d(DC[idx:(idx+1)].unsqueeze(0), kernel, padding='same').squeeze().cpu())
    return x_list, y_list

def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    if target_shape is None:
        return field
    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2
    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field

def compute_zernike_basis(num_polynomials, field_res):
    zernike_diam = int(np.ceil(np.sqrt(field_res[0]**2 + field_res[1]**2)))
    zernike = zernikeArray(num_polynomials, zernike_diam)
    zernike = crop_image(zernike, field_res, pytorch=False)
    zernike = torch.FloatTensor(zernike)
    return zernike

def preprocess_data(DC, true_gs):
    num = len(true_gs)
    x_list = []
    print('Loading into RAM')
    for idx in range(num):
      p_SLM_train = 2 * np.pi * torch.rand(1, 128, 128).float()
      if idx == num // 2:
        p_SLM_train = torch.zeros_like(p_SLM_train)
      x_train = torch.exp(1j * p_SLM_train)
      x_list.append(x_train)
    y_list = []
    print('Preprocessing y_batch')
    for idx in range(num):
      kernel = fftshift(fft2(true_gs[idx].unsqueeze(0) * x_list[idx].to(DC.device), norm="forward"), dim=[-2, -1]).abs() ** 2
      kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
      kernel = kernel.unsqueeze(0)
      y_list.append(F.conv2d(DC.unsqueeze(0), kernel, padding='same').squeeze().cpu())
    return x_list, y_list

# https://github.com/fkodom/fft-conv-pytorch
def complex_matmul(a, b, groups = 1):
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])

def fft_noPad_Conv2D(signal, kernel):
    signal_fr = rfftn(signal, dim=[-2, -1])
    kernel_fr = rfftn(kernel, dim=[-2, -1])

    output_fr = complex_matmul(signal_fr, kernel_fr)
    output = irfftn(output_fr, dim=[-2, -1], s=[-1, -1])
    output = ifftshift(output, dim=[-2, -1])

    return output

def fft_2xPad_Conv2D(signal, kernel, groups=1):
    size = signal.shape[-1]

    signal_fr = rfftn(signal, dim=[-2, -1], s=[2 * size, 2 * size])
    kernel_fr = rfftn(kernel, dim=[-2, -1], s=[2 * size, 2 * size])

    output_fr = complex_matmul(signal_fr, kernel_fr, groups)
    output = irfftn(output_fr, dim=[-2, -1], s=[-1, -1])
    s2 = size//2

    output = output[:, :, s2:-s2, s2:-s2]
    return output