# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import os, tqdm, math, imageio
from aotools.functions import zernikeArray
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.fft import fft2, fftshift, irfftn, rfftn, ifftshift

from PIL import Image
import matplotlib.pyplot as plt

ang_to_unit = lambda x : ((x / np.pi) + 1) / 2

def gen_moving_dataset(data_dir, num_frames=256):
    nums_per_image = 1
    original_size = 128
    width, height = 224, 224

    lims = (x_lim, y_lim) = width - original_size, height - original_size

    #direcs = np.pi * (np.random.rand(nums_per_image) * 2 - 1)
    direcs = np.array([0.97302908, -0.96160664])
    #speeds = (np.random.randint(5, size=nums_per_image) + 2) / 10
    speeds = np.array([0.2, 0.2]) * (128 / num_frames)

    veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])
    # Get a list containing two PIL images randomly sampled from the database
    mnist_images = [Image.open(f'{data_dir}/digit_6.png').convert('L').resize((original_size, original_size),Image.ANTIALIAS), 
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

    imageio.mimsave(f'{data_dir}/moving_n{num_frames}.gif', dataset, duration=1000*1./60)
    np.save(f'{data_dir}/moving_n{num_frames}.npy', dataset)

def preprocess_sim_dynamic_data(DC, true_gs):
    num = len(true_gs)
    x_list = []
    print('Loading into RAM')
    for idx in tqdm.trange(num):
      p_SLM_train = (torch.rand(1, DC.shape[-2], DC.shape[-1]).float() * 2 - 1) * np.pi
      #p_SLM_train = (torch.randn(1, DC.shape[-2], DC.shape[-1]).float())
      p_SLM_train = p_SLM_train
      if idx % 100 == 0:
        p_SLM_train = torch.zeros_like(p_SLM_train)
      x_train = torch.exp(1j * p_SLM_train)
      x_list.append(x_train)

    y_list = []
    print('Preprocessing y_batch')
    ys = []
    y_min, y_max = 100, 0
    for idx in tqdm.trange(num):
      true_g = true_gs[idx].unsqueeze(0)
      kernel = fftshift(fft2(true_g * x_list[idx].to(DC.device), norm="forward"), dim=[-2, -1]).abs() ** 2
      kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
      kernel = kernel.unsqueeze(0)
      kernel = kernel.flip(2).flip(3)

      y = F.conv2d(DC[idx:(idx+1)].unsqueeze(0), kernel, padding='same').squeeze().cpu()
      ys.append(y)
      if y.min() < y_min: y_min = y.min() 
      if y.max() > y_max: y_max = y.max() 
    print(f'y_min: {y_min}, y_max: {y_max}')
    for y in ys:
      normalized_y = (y - y_min) / (y_max - y_min)
      #normalized_y = y
      y_list.append(normalized_y)
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

    output_fr = signal_fr * kernel_fr
    #output_fr = complex_matmul(signal_fr, kernel_fr)
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

def preprocess_data(DC, true_gs):
    num = len(true_gs)
    x_list = []
    print('Preprocessing x_batch')
    for idx in range(num):
      #p_SLM_train = np.pi * (2 * torch.rand(1, DC.shape[-2], DC.shape[-1]).float() - 1)
      rand_patterns = 2 * torch.rand(1, 1, 4, 4).float() - 1
      size = (DC.shape[-2], DC.shape[-1])
      rand_patterns = tv.transforms.Resize(size)(rand_patterns)
      p_SLM_train = np.pi * rand_patterns.squeeze(0)
      if idx == -1:
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


import scipy.io as sio
import torchvision
def preprocess_data2(DC, true_gs, width=128):
    num = len(true_gs)
    x_list = []
    data_dir = 'data/0725'
    slm_prefix = 'SLM_sim'
    print('Preprocessing x_batch')
    resize = torchvision.transforms.Resize((width, width), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    for idx in range(num):
        mat_name = f'{data_dir}/{slm_prefix}{idx+1}.mat'
        p_SLM = sio.loadmat(f'{mat_name}')
        p_SLM = p_SLM['proj_sim']
        if idx == 0:
            aperture = np.ones_like(p_SLM)
            aperture = np.lib.pad(aperture, (((256 - 144) // 2, (256 - 144) // 2), (0, 0)), 'constant', constant_values=(0, 0))
            aperture = torch.FloatTensor(aperture).unsqueeze(0)

        p_SLM = np.lib.pad(p_SLM, (((256 - 144) // 2, (256 - 144) // 2), (0, 0)), 'constant', constant_values=(0, 0))
        p_SLM_train = torch.FloatTensor(p_SLM).unsqueeze(0)
        p_SLM_train = resize(p_SLM_train)
        aperture = resize(aperture)
        x_train = aperture * torch.exp(1j * p_SLM_train)
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

def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()) for w in weights]

def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)

def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.
        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())

def create_random_direction(net, dir_type, ignore='biasbn', norm='filter'):
    # random direction
    weights = get_weights(net) # a list of parameters.
    direction = get_random_weights(weights)
    direction = [d.to("cuda") for d in direction]
    normalize_directions_for_weights(direction, weights, norm, ignore)

    return direction

def setup_directions(net, dir_type='weights', xignore='biasbn', xnorm='filter'):
    xdirection = create_random_direction(net, dir_type, xignore, xnorm)
    ydirection = create_random_direction(net, dir_type, xignore, xnorm)

    return [xdirection, ydirection]

def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    dx, dy = directions[0], directions[1]
    changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
    for (p, w, d) in zip(net.parameters(), weights, changes):
        p.data = w + torch.Tensor(d).type(type(w)).to(w.device)

def write_dir(dir_file, xdirection, ydirection):
    f = h5py.File(dir_file,'w')
    write_list(f, 'xdirection', xdirection)
    write_list(f, 'ydirection', ydirection)

def write_list(f, name, direction):
    """ Save the direction to the hdf5 file with name as the key
        Args:
            f: h5py file object
            name: key name_surface_file
            direction: a list of tensors
    """

    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, torch.Tensor):
            l = l.cpu().numpy()
        grp.create_dataset(str(i), data=l)

def read_list(f, name):
    """ Read group with name as the key from the hdf5 file and return a list numpy vectors. """
    grp = f[name]
    return [grp[str(i)] for i in range(len(grp))]
