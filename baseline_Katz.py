import scipy.io as sio
import itertools, logging, warnings
import argparse
from baseline_utils import *
from piq import psnr
from scipy import signal
import shutil
import time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_name', default='d1-48', type=str)
    parser.add_argument('--gaussian_std', default=None, type=int)
    args = parser.parse_args()
    img_name  = args.img_name
    gaussian_std = args.gaussian_std

    # static parameters
    save_root_folder = './baseline_outputs'
    save_freq = 1000
    slm_coarse_size = 2
    coarse2fine = True
    decay_factor = 1000 # 5000
    refine_traceback_iter = 1000 # 10000
    final_traceback_iter = 1000 # 20000
    min_refine_iter = 1e3
    final_refine_iter = 1e5 # 1e3
    traceback_ratio = 0.001  # 0.00001
    mu_ratio_initial = .36  # .3
    mu_ratio_final = .013 # .013
    alpha = 1.0 # 0.4
    n_iter = {'Entropy': 5e5, 'Variance': int(1e5), 'PSNR': 100000}
    mode_list = ['Variance']
    width = 64
    seed = 0
    circ_gaussian = True
    Dr0 = 1
    gaussian_size = 7
    extra_name =  f'Baseline-{img_name}-Size{gaussian_size}-STD{gaussian_std}-Oris'
    hereditary_ratio = 10 # not in use! for sigmoid parent indice
    slm_range = 51 # random turbulence
    mutate_gaussian = False
    mutate_range = 6 # only in-use when mutate_mode is gaussian
    n_slm = 128
    data_dir = 'data'
    PSF_size = width
    num_worker = 1
    n_parents = int(0.5 * n_slm)
    n_child = n_slm - n_parents
    shrink_tb = False
    tb_size = 256 # only in-use when shrink_tb is set to True

    # settings
    seed_torch(0)
    warnings.filterwarnings("ignore")
    device = 'cuda'
    out_dir = f'{save_root_folder}/{extra_name}'
    print(out_dir)
    if circ_gaussian:
        noise_mode = 'Scatter'
    else:
        noise_mode = 'Zernike'
    shutil.rmtree(os.path.abspath(out_dir), ignore_errors=True)
    os.makedirs(out_dir)
    os.makedirs(f'{out_dir}/iters')
    os.makedirs(f'{out_dir}/figs')
    os.makedirs(f'{out_dir}/codes')
    os.makedirs(f'{out_dir}/selected')
    shutil.copy(f'{os.getcwd()}/ori.py', f'{out_dir}/codes')
    shutil.copy(f'{os.getcwd()}/utils.py', f'{out_dir}/codes')
    shutil.copy(f'{os.getcwd()}/myutils.py', f'{out_dir}/codes')
    special_it_indice = []
    special_corr_list = []
    for i in range(100):
        special_it_indice.append(2**i)

    # Load in data
    I_obj = np.asarray(Image.open(f'data/{img_name}.png').convert('L').resize((width, width))) / 255.
    truth = transforms.Resize([width, width])(torch.tensor(I_obj).unsqueeze(0)).float().to(device)
    imageio.imsave(f'{out_dir}/truth.png', np.uint8(255 * truth.detach().cpu().numpy()).squeeze())
    z_basis = compute_zernike_basis(num_polynomials=28, field_res=(PSF_size, PSF_size)).permute(1, 2, 0).to(device)
    temp = compute_zernike_basis(num_polynomials=4, field_res=(3, 3))

    # sim turbulence
    if circ_gaussian:
        _, tb = gen_true_turbulence(None, 1, z_basis, out_dir, PSF_size=PSF_size, device=device,
                                    out_squeeze=True, Dr0=Dr0, gaussian_std=gaussian_std,
                                    gaussian_size=gaussian_size, no_tilt=True, circ_gaussian=circ_gaussian)
    else:
        tb, _ = gen_true_turbulence(None, 1, z_basis, out_dir, PSF_size=PSF_size, device=device,
                                    out_squeeze=True, Dr0=Dr0, gaussian_std=gaussian_std,
                                    gaussian_size=gaussian_size, no_tilt=True, circ_gaussian=circ_gaussian)
    if shrink_tb:
        tb = transforms.Resize(tb_size, interpolation=transforms.InterpolationMode.BILINEAR)(tb.unsqueeze(0))
        tb = transforms.Resize(width, interpolation=transforms.InterpolationMode.NEAREST)(tb.unsqueeze(0)).squeeze()

    sample_no_slm = gen_measurements(truth, tb, torch.zeros(1, 1, 1).to(device), width, circ_gaussian=circ_gaussian, out_numpy=False, device=device).to(device)
    imageio.imsave(f'{out_dir}/measurement.png', np.uint8(255 * normalize(sample_no_slm).detach().cpu().numpy()).squeeze())


    # Generate SLM parents and children
    slm_parents = (2 * slm_range * torch.rand(n_parents, slm_coarse_size, slm_coarse_size) - slm_range).float().to(device)
    slm_child = (2 * slm_range * torch.rand(n_child, slm_coarse_size, slm_coarse_size) - slm_range).float().to(device)
    y_parents = gen_measurements(truth, tb, slm_parents, width, circ_gaussian=circ_gaussian, out_numpy=False, device=device).to(device)
    truth_repeated = truth.unsqueeze(0).repeat(n_slm, 1, 1, 1)
    parent_permutation = list(itertools.permutations(iterable=np.arange(n_parents), r=2))
    parent_permutation = np.stack([np.array([x[0], x[1]]) for x in parent_permutation if x[0] < x[1]])

    for mode in mode_list:
        metric_mean_list = []
        tqdm_bar = tqdm(range(int(n_iter[mode])), desc='', disable=False)
        last_best_metric = -1e10
        last_best_iter = 0

        for iter in tqdm_bar:
            rest_start = time.time()
            mutation_ratio = (mu_ratio_initial - mu_ratio_final) * torch.exp(torch.tensor(-len(metric_mean_list)/decay_factor)) + mu_ratio_final
            if iter == 1:
                print()

            # gen measurements for children
            y_child = gen_measurements(truth, tb, slm_child, width, circ_gaussian=circ_gaussian, out_numpy=False, device=device)
            y_batch = torch.cat((y_parents, y_child), 0)
            slm_batch = torch.cat((slm_parents, slm_child), 0).to(device)

            # calculate score
            if mode == 'Entropy':
                y_scaled = alpha * y_batch / torch.amax(y_batch, dim=(1, 2), keepdim=True)
                metric = torch.sum(y_scaled * torch.log(y_scaled), dim=(1, 2), keepdim=False) # entropy (maximize)  here it is the entropy without the negative sign (opposite from paper)
            elif mode == 'Variance':
                metric = torch.var(y_batch.view(n_slm, -1), dim=1, keepdim=False)     # variance (maximize)
            elif mode == 'PSNR':
                metric = psnr(torch.clamp(y_batch.unsqueeze(1), 0, 1), truth_repeated, reduction='none')
            else:
                print('mode must be entropy or variance or PSNR or TV. exit...'); exit;

            sorted_score, sorted_indice = metric.topk(n_slm, largest=True, sorted=True)

            # selected higher-ranked slm patterns as parents with more probability
            parent_probs = np.linspace(1, 0, n_slm - 1, endpoint=False)
            parent_probs = 1 / (1 + np.exp(-(parent_probs - 0.5) * hereditary_ratio)) * 1.0
            parent_probs = parent_probs / np.sum(parent_probs)
            parent_indice = np.random.choice(sorted_indice[1:].cpu().detach().numpy(), n_parents - 1, p=parent_probs, replace=False)
            parent_indice = np.concatenate((np.array([sorted_indice[0].cpu().detach().numpy()]), parent_indice), 0)

            # save results
            if sorted_score[0].item() > last_best_metric:
                last_best_metric = sorted_score[0].item()
                last_best_iter = iter

            slm_best = slm_batch[sorted_indice[0]]
            recon_best = y_batch[sorted_indice[0]]


            if iter in special_it_indice:
                metric = crosscorr(recon_best, truth)
                imageio.imsave(f'{out_dir}/selected/Ori_NumT_{int(iter*n_slm)}_{metric}.png', np.uint8(255 * normalize(recon_best).detach().cpu().numpy()).squeeze())

            if not circ_gaussian:
                psnr_value = psnr(torch.clamp(recon_best[None, None, ...], 0, 1), truth[None, ...])
            else:
                psnr_value = 0

            tqdm_bar.set_description(f'[Iter {iter}]: SLM GridSize {slm_coarse_size} {mode} {sorted_score[0].item():.14f} (from iter {last_best_iter}) Metric {psnr_value:.4f} Mutation Rate {mutation_ratio}')
            tqdm_bar.refresh()


            if iter % save_freq == 0:
                imageio.imsave(f'{out_dir}/recon.png', np.uint8(255 * normalize(recon_best).detach().cpu().numpy()).squeeze())
                imageio.imsave(f'{out_dir}/iters/recon_{mode}_{iter}.png', np.uint8(255 * recon_best.detach().cpu().numpy()).squeeze())
                best_psnr = crosscorr(recon_best, truth)

                for corr_threshold in special_corr_list:
                        imageio.imsave(f'{out_dir}/{extra_name}_Ori_{corr_threshold}_NumNeeded_{int(iter*n_slm)}.png', np.uint8(255 * recon_best.detach().cpu().numpy()).squeeze())
                        special_corr_list.remove(corr_threshold)

            # decrease SLM size
            metric_mean_list.append(torch.mean(sorted_score).item())
            if coarse2fine and len(metric_mean_list) > min_refine_iter and abs(metric_mean_list[-1] - metric_mean_list[-refine_traceback_iter]) < abs(metric_mean_list[0]) * traceback_ratio and slm_coarse_size < width:
                slm_coarse_size = min(slm_coarse_size * 2 , width)
                slm_parents = transforms.Resize(slm_coarse_size, interpolation=transforms.InterpolationMode.NEAREST)(slm_parents)
                slm_child = transforms.Resize(slm_coarse_size, interpolation=transforms.InterpolationMode.NEAREST)(slm_child)
                slm_batch = transforms.Resize(slm_coarse_size, interpolation=transforms.InterpolationMode.NEAREST)(slm_batch)
                del metric_mean_list
                metric_mean_list = []

            # parents breed
            y_parents = y_batch[parent_indice, ...].to(device)
            slm_parents = slm_batch[parent_indice, ...].to(device)
            masks = torch.rand(n_child, slm_child.shape[1], slm_child.shape[2]) > 0.5
            masks = masks.to(device)
            parent_pairs = random.sample(range(len(parent_permutation)), int(n_child))
            slm_moms = slm_parents[parent_permutation[parent_pairs][:, 0]]
            slm_dads = slm_parents[parent_permutation[parent_pairs][:, 1]]
            slm_child = slm_moms * masks + slm_dads * ~masks
            rest_time = time.time() - rest_start

            # mutate child
            sample_start = time.time()
            mutated_pixel_idx = torch.randperm(len(slm_child.flatten()))[:int(len(slm_child.flatten()) * mutation_ratio)].to(device)
            sample_time = time.time() - sample_start
            slm_child_flatten = slm_child.view(len(slm_child.flatten())).to(device)

            mutate_start = time.time()
            if not mutate_gaussian:
                slm_child_flatten[mutated_pixel_idx] = (2 * slm_range * torch.rand(len(mutated_pixel_idx)) - slm_range).float().to(device)
            else:
                slm_child_flatten[mutated_pixel_idx] = torch.clamp(torch.normal(mean=slm_child_flatten[mutated_pixel_idx], std=mutate_range), min=-slm_range, max=slm_range)

            mutate_time = time.time() - mutate_start
            slm_child = slm_child_flatten.view(n_child, slm_child.shape[1], slm_child.shape[2])
            if iter == n_iter[mode] - 1:
                del metric_mean_list
                metric_mean_list = []

    sio.savemat(f'{out_dir}/final_slm.mat', {'data': slm_best})


