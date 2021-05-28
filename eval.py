import h5py
import argparse
import os
import torch
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
from trainer import Trainer
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import matplotlib
import seaborn as sns

parser = argparse.ArgumentParser(description='SAILER')
parser.add_argument('--name', default='main', type=str, help='name of the experiment')
parser.add_argument('--log', default='train_log.csv', type=str, help='name of log file')
parser.add_argument('-l', '--load_ckpt', default=False, type=str, help='path to ckpt loaded')
parser.add_argument('-cuda', '--cuda_dev', default=[0], type=list, help='GPU want to use')
parser.add_argument('--sample_batch', default=False, type=bool, help='Add batch effect correction')
parser.add_argument('--max_epoch', default=400, type=int, help='maximum training epoch')
parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch')
parser.add_argument('-b', '--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--start_save', default=350, type=int, help='epoch starting to save models')
parser.add_argument('--conv', default=False, type=bool, help='use conv vae')
parser.add_argument('--model_type', default='inv', type=str, help='model type')
parser.add_argument('-d', '--data_type', type=str, help='dataset')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--pos_w', default=30, type=float, help='BCE positive weight')
parser.add_argument('--weight_decay', default=5e-4, type=str, help='weight decay for adam')
parser.add_argument('--z_dim', default=10, type=int, help='latent dim')
parser.add_argument('--out_every', default=1, type=int, help='save ckpt every x epoch')
parser.add_argument('--ckpt_dir', default='./models/', type=str, help='out directory')
parser.add_argument('--setting', default=2, type=int, help='setting for sim data')
parser.add_argument('--signal', default=0.35, type=float, help='signal to noise ratio')
parser.add_argument('--frags', default=3000, type=int, help='num of fragments')
parser.add_argument('--bin_size', default=10000, type=int, help='size of each bin')
parser.add_argument('--LAMBDA', default=1, type=float, help='lambda value')
args = parser.parse_args()

solver = Trainer(args)
latent, labels, depth = solver.encode_adv()
result = latent.numpy()
l = pd.DataFrame(labels, columns=['celltype'])
font = {'family': 'Times New Roman','size': 15}

def get_cmap1(n, name='tab20b'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_cmap2(n, name='tab20c'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

cmap1 = get_cmap1(20)
cmap2 = get_cmap2(20)

print('tsne...')
X_embedded = TSNE(n_components=2, perplexity=50).fit_transform(result)
plt.figure(figsize=(8, 8))
i = 0
for c in np.unique(l):
    mask = (l == c).values.flatten()
    if i<20:
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=c, s=3, color=cmap1(i))
    else:
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=c, s=3, color=cmap2(i-20))
    i += 1
    plt.xticks([], [])
    plt.yticks([], [])
plt.tight_layout()
plt.savefig('./results/TSNE_'+ args.name + '.png', dpi=300)