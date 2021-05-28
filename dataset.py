import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import h5py
import time
from scipy.sparse import load_npz, vstack

import torch
from torch.utils.data import Dataset, DataLoader

data_path3 = './data/SimATAC/'
data_path5 = './data/MouseAtalas/'
data_path7 = './data/PBMC/'

Downsample = 1000

class SimATAC_peak(Dataset):
    def __init__(self, setting=1, signal=0.8, frags=5000, bin_size=None, data_pth=data_path3, conv=False, downsample=Downsample):
        super(SimATAC_peak, self).__init__()
        file_name = f'setting{setting}_s{signal}_f{frags}.SCAN-ATAC-Sim.npz'
        type_name = f'setting{setting}_s{signal}_f{frags}.SCAN-ATAC-Sim_labels.txt'
        tic = time.time()
        self.data = load_npz(os.path.join(data_path3, file_name))
        self.cell_label = list(pd.read_table(os.path.join(data_path3, type_name), sep='\t', header=None)[1].values.flatten())
        self.size = self.data.shape[-1]
        self.conv = conv
        tok = time.time()
        print(f"Finish loading in {tok-tic}, data size {self.data.shape}")
        if conv:
            self.padto = (int(self.size/downsample) + 1) * downsample
        else:
            self.padto = self.size
        self.depth_data = self.data.copy()
        self.d_mean = np.log(self.depth_data.sum(axis=1)).mean()
        self.d_std = np.log(self.depth_data.sum(axis=1)).std()
        # self.d_mean = self.data.sum(axis=1).mean()
        # self.d_std = self.data.sum(axis=1).std()

    def __getitem__(self, index):
        if self.conv:
            y = np.zeros(self.padto)
            x = np.array(self.data[index].todense()).flatten().astype(np.bool).astype(np.float)
            y[:x.shape[0]] = x
            return y, self.cell_label[index], np.sum(x)
        else:
            x = np.array(self.data[index].todense()).flatten().astype(np.bool).astype(np.float)
            return x, self.cell_label[index], np.sum(x)

    def __len__(self):
        return len(self.cell_label)


class MouseAtlas(Dataset):
    def __init__(self, data_pth=data_path5, cutoff=None):
        super(MouseAtlas, self).__init__()
        file_name = 'atlas_dataset_50k_int.npz'
        type_name = 'mouse_atlas_label.txt'
        tic = time.time()
        self.data = load_npz(os.path.join(data_pth, file_name))
        self.cell_label = list(pd.read_table(os.path.join(data_pth, type_name), header=None)[1].values.flatten())
        if cutoff is not None:
            select = np.random.choice(range(self.data.shape[0]), size=cutoff)
            self.data = self.data[select]
            self.data = vstack((self.data, self.data ))
            self.cell_label = np.array(self.cell_label)[select]
            self.cell_label = np.hstack((self.cell_label, self.cell_label))
        self.size = self.data.shape[-1]
        tok = time.time()
        print(f"Finish loading in {tok-tic}, data size {self.data.shape}")
        self.padto = self.size
        self.d_mean = np.log(self.data.sum(axis=1)).mean()
        self.d_std = np.log(self.data.sum(axis=1)).std()


    def __getitem__(self, index):
        x = np.array(self.data[index].todense()).flatten().astype(np.bool).astype(np.float)
        return x, self.cell_label[index], np.sum(x)

    def __len__(self):
        return len(self.cell_label)


class PBMC(Dataset):
    def __init__(self, data_pth=data_path7):
        super(PBMC, self).__init__()
        file_name = 'merge_matrix_70k.npz'
        type_name = 'batch_label.txt'
        tic = time.time()
        self.data = load_npz(os.path.join(data_pth, file_name))
        self.cell_label = list(pd.read_table(os.path.join(data_pth, type_name))['label'].values.flatten())
        self.size = self.data.shape[-1]
        tok = time.time()
        print(f"Finish loading in {tok-tic}, data size {self.data.shape}")
        self.padto = self.size
        self.d_mean = np.log(self.data.sum(axis=1)).mean()
        self.d_std = np.log(self.data.sum(axis=1)).std()

    def __getitem__(self, index):
        x = np.array(self.data[index].todense()).flatten().astype(np.bool).astype(np.float)
        return x, self.cell_label[index], np.sum(x)

    def __len__(self):
        return len(self.cell_label)


class MergeSim(Dataset):
    def __init__(self, num='', data_pth=data_path3):
        super(MergeSim, self).__init__()
        file_name = f'merge_sim{num}.npz'
        type_name = f'merge_sim_labels{num}.csv'
        tic = time.time()
        self.data = load_npz(os.path.join(data_pth, file_name))
        self.cell_label = list(pd.read_csv(os.path.join(data_pth, type_name), header=None)[1].values.flatten())
        self.size = self.data.shape[-1]
        tok = time.time()
        print(f"Finish loading in {tok-tic}, data size {self.data.shape}")
        self.padto = self.size
        self.d_mean = np.log(self.data.sum(axis=1)).mean()
        self.d_std = np.log(self.data.sum(axis=1)).std()

    def __getitem__(self, index):
        x = np.array(self.data[index].todense()).flatten().astype(np.bool).astype(np.float)
        return x, self.cell_label[index], np.sum(x)

    def __len__(self):
        return len(self.cell_label)
