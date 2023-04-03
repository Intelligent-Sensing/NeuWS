import os
import torch
import numpy as np
import scipy.io as sio
import imageio

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, x_list, y_list):
        self.x_list, self.y_list = x_list, y_list

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx], idx

class BatchDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, im_prefix='SLM_raw', slm_prefix='SLM_sim', num=100, max_intensity=30000):
        self.data_dir = data_dir
        a_slm = np.ones(sio.loadmat(f'{self.data_dir}/SLM_sim1.mat')['proj_sim'].shape)
        a_slm = np.lib.pad(a_slm, (((256 - 144) // 2, (256 - 144) // 2), (0, 0)), 'constant', constant_values=(0, 0))
        self.a_slm = torch.from_numpy(a_slm).type(torch.float)
        self.max_intensity = max_intensity
        self.num = num
        self.load_in_cache()
        self.num = len(self.xs)
        self.im_prefix, self.slm_prefix = im_prefix, slm_prefix

    def load_in_cache(self):
        x_list, y_list = [], []
        for idx in range(self.num):
            #img_name = f'{self.data_dir}/exp_raw{idx+1}.mat'
            img_name = f'{self.data_dir}/{self.im_prefix}{idx+1}.mat'
            mat_name = f'{self.data_dir}/{self.slm_prefix}{idx+1}.mat'

            p_SLM = sio.loadmat(f'{mat_name}')
            p_SLM = p_SLM['proj_sim']
            p_SLM = np.lib.pad(p_SLM, (((256 - 144) // 2, (256 - 144) // 2), (0, 0)), 'constant', constant_values=(0, 0))
            p_SLM_train = torch.FloatTensor(p_SLM).unsqueeze(0)
            x_train = self.a_slm * torch.exp(1j * -p_SLM_train)

            ims = sio.loadmat(f'{img_name}')
            y_train = ims['imsdata']
            y_train = torch.FloatTensor(y_train)
            y_train = y_train / self.max_intensity
            #print(f'{mat_name}: {x_train.shape, y_train.shape}')
            x_list.append(x_train); y_list.append(y_train)
        self.xs, self.ys = x_list, y_list

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], idx