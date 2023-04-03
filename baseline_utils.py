import os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift
import shutil, imageio
from torchvision import transforms
import torchgeometry as tgm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision
import piq.psnr as piq_psnr
import piq.ssim as piq_ssim
from torch.fft import fft2, fftshift, irfftn, rfftn, ifftshift
from scipy import signal
from aotools.functions import zernikeArray


ang_to_unit = lambda x : ((x / np.pi) + 1) / 2


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


def gen_true_turbulence(DC, num_t, z_basis, out_dir, PSF_size=64, device='cuda', out_numpy=False, out_squeeze=False, Dr0=1,
                        retun_zern_coeff=False, downsample_scale=1, gaussian_std=None, gaussian_size=None, no_tilt=False, circ_gaussian=False):
    seed_torch(0)
    if circ_gaussian:

        real_part = torch.randn(PSF_size, PSF_size).float() * Dr0
        imag_part = torch.randn(PSF_size, PSF_size).float() * Dr0
        if gaussian_std is not None:
            real_part = tgm.image.gaussian_blur(real_part[None, None, ...], (gaussian_size, gaussian_size), (gaussian_std, gaussian_std)).squeeze()
            imag_part = tgm.image.gaussian_blur(imag_part[None, None, ...], (gaussian_size, gaussian_size), (gaussian_std, gaussian_std)).squeeze()

        abe_0 = (1 / np.sqrt(2)) * torch.complex(real_part, imag_part)
        abe_0 = abe_0.to(device)
        return None, abe_0

    z_basis = z_basis.to(device)
    delta = np.array([ 1.1416363 , -0.43582536, -0.06260428,  0.38009766,  0.25361097,
                       -0.3148466 ,  0.92747134,  1.68069706, -2.32103207,  2.64558499,
                       -1.35933578,  2.26350006,  0.47954541,  1.86205627,  0.23382413,
                       -2.45344533, -0.90342144, -1.18094509, -2.21078039, -0.80214435,
                       2.43905707,  0.38972859, -0.14482323,  0.57469502, -0.37923575,
                       0.67929707, -0.37997045,  1.7486404 ])
    alpha_0 = 2 * torch.FloatTensor([[-0.05455], [-0.04884], [-0.14921], [1.76161], [-2.18389], [3.22574], [-3.12588], [0.04051],
                                     [-2.15437], [3.27762], [-3.45384], [2.37893], [-2.00064], [3.45384], [-2.37893], [0.62289], [-2.18389], [-3.22574], [-3.12588], [-0.04051],
                                     [3.45384], [-2.37893], [0.62289], [-2.00064], [0.62289], [-2.00064], [2.08325], [-1.76161]]).to(device)

    alpha_1 = alpha_0 - torch.FloatTensor(delta).unsqueeze(1).to(device) * 1.2
    alpha_0 *= Dr0
    alpha_1 *= Dr0

    if retun_zern_coeff:
        return alpha_1
    coeffs = []
    out_frames = []
    true_gs = []
    true_gs_np, coeffs_np = [], []
    for i, t in enumerate(np.linspace(start=0, stop=1, num=num_t, endpoint=True)):
        temp = t * alpha_1 + (1 - t) * alpha_0
        if no_tilt:
            temp[:3, :] = 0
        coeff = (z_basis @ temp).squeeze(2)

        if gaussian_std is not None:
            coeff = tgm.image.gaussian_blur(coeff[None, None, ...], (gaussian_size, gaussian_size), (gaussian_std, gaussian_std)).squeeze()

        g = torch.exp(1j * coeff)
        true_gs_np.append(g.cpu().detach().numpy())
        coeffs_np.append(coeff.cpu().detach().numpy())
        true_gs.append(g)
        coeffs.append(coeff)
        ave_coeffs = torch.mean(torch.stack(coeffs, 0))
        out_frames.append(np.uint8(255 * ang_to_unit(np.angle(g.cpu().detach().numpy()))))

    if out_numpy:
        return coeffs_np, true_gs_np

    if out_squeeze:
        return coeffs[-1], true_gs[-1]

    return coeffs, true_gs


def gen_measurements_gaussian(DC, tb, slm_list, width, input_zern_coeff=False, z_basis=None, save_dir=None, out_numpy=True, device='cuda'):
    # phase_sum input size 2D, ex. [128, 128]

    if input_zern_coeff:
        tb = (z_basis @ tb).squeeze(2)
        slm_list = (z_basis @ slm_list).permute(2, 0, 1)

    if not input_zern_coeff:
        if slm_list.size()[-1] != width:
            slm_list = transforms.Resize([width, width], interpolation=transforms.InterpolationMode.NEAREST)(slm_list)

    _kernel = (fftshift(fft2(tb.unsqueeze(0).unsqueeze(0) * torch.exp(1j * (slm_list.unsqueeze(1))), norm="forward"), dim=[-2, -1]).abs() ** 2)
    _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)
    out = F.conv2d(DC.unsqueeze(0), _kernel, padding='same').squeeze(0)

    return out


def gen_measurements(DC, tb, slm_list, width, circ_gaussian=False, input_zern_coeff=False, z_basis=None, save_dir=None, out_numpy=True, device='cuda'):
    # phase_sum input size 2D, ex. [128, 128]

    if input_zern_coeff:
        tb = (z_basis @ tb).squeeze(2)
        slm_list = (z_basis @ slm_list).permute(2, 0, 1)

    if not input_zern_coeff:
        if slm_list.size()[-1] != width:
            slm_list = transforms.Resize([width, width], interpolation=transforms.InterpolationMode.NEAREST)(slm_list)

    if not circ_gaussian:
        _kernel = (fftshift(fft2(torch.exp(1j * (tb.unsqueeze(0).unsqueeze(0) + slm_list.unsqueeze(1))), norm="forward"), dim=[-2, -1]).abs() ** 2)
    else:
        _kernel = (fftshift(fft2(torch.exp(1j * slm_list.unsqueeze(1)) * tb.unsqueeze(0).unsqueeze(0), norm="forward"), dim=[-2, -1]).abs() ** 2)

    _kernel = _kernel / torch.sum(_kernel, dim=[-2, -1], keepdim=True)

    _kernel = _kernel.flip(2).flip(3)
    if not circ_gaussian:
        out = F.conv2d(DC.unsqueeze(0), _kernel, padding='same').squeeze(0)
    else:
        out = fft_2xPad_Conv2D(DC.unsqueeze(0), _kernel).squeeze(0)

    return out


    if save_dir is not None:
        imageio.mimsave(f'{save_dir}', y_list, fps=60)

    if out_numpy:
        return np.stack(y_list)

    return torch.stack(y_list)


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def psnr_translation_invariant(corrected, truth, pad_margin=10, metric='psnr', istorch=True, normalize=True, verbose=False, device='cuda'):
    """

    if torch, input is [B, C, H, W];
    if numpy, input is [H, W]
    """

    if istorch:

        if normalize:
            truth = (truth - torch.min(truth)) / (torch.max(truth) - torch.min(truth))
            corrected = (corrected - torch.min(corrected)) / (torch.max(corrected) - torch.min(corrected))

        truth_padded = torchvision.transforms.Pad(pad_margin//2)(truth).float().to(device)

        best_psnr = 0
        best_h = 0
        for h in range(pad_margin):
            for w in range(pad_margin):
                corrected_padded = torchvision.transforms.Pad((w, h, pad_margin-w, pad_margin-h))(corrected).float().to(device)
                # corrected_padded = np.pad(corrected, ((h, pad_margin-h), (w, pad_margin-w)), mode='constant', constant_values=0)

                if 'psnr' in metric.lower():
                    psnr = piq_psnr(torch.clamp(corrected_padded, 0, 1), torch.clamp(truth_padded, 0, 1))
                if 'ssim' in metric.lower():
                    psnr = piq_ssim(torch.clamp(corrected_padded, 0, 1), torch.clamp(truth_padded, 0, 1))
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_h = h
        if verbose:
            print('best psnr: ', best_psnr, ', best h: ', best_h, ', total h: ', 'best')

    else:
        truth_padded = np.pad(truth, ((pad_margin//2, pad_margin//2), (pad_margin//2, pad_margin//2)), mode='constant', constant_values=0)

        if normalize:
            truth_padded = (truth_padded - np.min(truth_padded)) / (np.max(truth_padded) - np.min(truth_padded))  * 255
            corrected = (corrected - np.min(corrected)) / (np.max(corrected) - np.min(corrected)) * 255

        best_psnr = 0
        best_h = 0
        for h in range(pad_margin):
            for w in range(pad_margin):
                corrected_padded = np.pad(corrected, ((h, pad_margin-h), (w, pad_margin-w)), mode='constant', constant_values=0)

                if 'psnr' in metric.lower():
                    psnr = peak_signal_noise_ratio(corrected_padded, truth_padded)
                if 'ssim' in metric.lower():
                    psnr = structural_similarity(corrected_padded, truth_padded)

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_h = h
        if verbose:
            print('best psnr: ', best_psnr, ', best h: ', best_h, ', total h: ', 'best')

    return best_psnr


def genTarget(unpaddedWidth=16, totalWidth=16, info_ratio=0.25, device='cuda', seed=0):

    seed_torch(seed)
    DC = torch.arange(unpaddedWidth**2)
    DC = DC / len(DC)
    DC = DC < info_ratio
    DC = DC[torch.randperm(len(DC))]
    DC = DC.float().resize(unpaddedWidth, unpaddedWidth).to(device)
    DC = transforms.Pad(int((totalWidth - unpaddedWidth)/2))(DC[None, None, ...])
    return DC.squeeze(0)

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


def fft_2xPad_Conv2D(signal, kernel):
    size = signal.shape[-1]

    signal_fr = rfftn(signal, dim=[-2, -1], s=[2 * size, 2 * size])
    kernel_fr = rfftn(kernel, dim=[-2, -1], s=[2 * size, 2 * size])

    output_fr = complex_matmul(signal_fr, kernel_fr)
    output = irfftn(output_fr, dim=[-2, -1], s=[-1, -1])
    s2 = size//2
    output = output[:, :, s2:-s2, s2:-s2]

    return output

def normalize(vector):
    return (vector - torch.min(vector))/(torch.max(vector)-torch.min(vector))


def crosscorr(v1, v2):
    return np.max(signal.correlate2d(v1.squeeze().detach().cpu().numpy(), v2.squeeze().detach().cpu().numpy(),"full"))/np.linalg.norm(v1.squeeze().detach().cpu().numpy())/np.linalg.norm(v2.squeeze().detach().cpu().numpy())
