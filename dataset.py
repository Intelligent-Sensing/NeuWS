# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import torch
import numpy as np
import scipy.io as sio

class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, im_prefix='SLM_raw', slm_prefix='SLM_sim', num=100, max_intensity=0, zero_freq=-1):
        self.data_dir = data_dir
        self.zero_freq = zero_freq
        a_slm = np.ones((144, 256))
        a_slm = np.lib.pad(a_slm, (((256 - 144) // 2, (256 - 144) // 2), (0, 0)), 'constant', constant_values=(0, 0))
        self.a_slm = torch.from_numpy(a_slm).type(torch.float)
        self.max_intensity = max_intensity
        self.num = num
        self.im_prefix, self.slm_prefix = im_prefix, slm_prefix
        self.load_in_cache()
        self.num = len(self.xs)
        print(f'Training with {self.num} frames.')

    def load_in_cache(self):
        x_list, y_list = [], []
        for idx in range(self.num):
            img_name = f'{self.data_dir}/{self.im_prefix}{idx+1}.mat'
            mat_name = f'{self.data_dir}/{self.slm_prefix}{idx+1}.mat'

            try:
                p_SLM = sio.loadmat(f'{mat_name}')
                p_SLM = p_SLM['proj_sim']
                p_SLM = np.lib.pad(p_SLM, (((256 - 144) // 2, (256 - 144) // 2), (0, 0)), 'constant', constant_values=(0, 0))
                p_SLM_train = torch.FloatTensor(p_SLM).unsqueeze(0)

                if self.zero_freq > 0 and idx % self.zero_freq == 0:
                    p_SLM_train = torch.zeros_like(p_SLM_train)
                    img_name = f'{self.data_dir}/../Zero/{self.im_prefix}{idx+1}.mat'
                    print(f'#{idx} uses zero SLM')

                x_train = self.a_slm * torch.exp(1j * -p_SLM_train)
                ims = sio.loadmat(f'{img_name}')
                y_train = ims['imsdata']

                if np.max(y_train) > self.max_intensity:
                    self.max_intensity = np.max(y_train)

                y_train = torch.FloatTensor(y_train)
                x_list.append(x_train); y_list.append(y_train)

            except Exception as e:
                print(f'{e}')
                continue
        y_list = [y / self.max_intensity for y in y_list]
        self.xs, self.ys = x_list, y_list

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], idx


