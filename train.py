import argparse
from trainer import Trainer
import torch
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='SAILER')
parser.add_argument('--name', default='main', type=str, help='name of the experiment')
parser.add_argument('--log', default='train_log.csv', type=str, help='name of log file')
parser.add_argument('--load_ckpt', default=False, type=str, help='path to ckpt loaded')
parser.add_argument('-cuda', '--cuda_dev', default=[0], type=list, help='GPU want to use')
parser.add_argument('--sample_batch', default=False, type=bool, help='Add batch effect correction')
parser.add_argument('--max_epoch', default=400, type=int, help='maximum training epoch')
parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch')
parser.add_argument('-b', '--batch_size', type=int, help='batch size')
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
parser.add_argument('--lambda', default=1, type=float, help='lambda value')



# args = EZDict({
#     "name": 'invvae_lambda5_atlas', #'invvae_bce30_logznorm_atlas', #"warmup_train_s0.8f5000bs10k"
#     'log': 'train_log.csv',
#     'load_ckpt': '/home/yingxinc4/DeepATAC/models/invvae_lambda5_atlas/400.pt',
#     'cuda_dev': [2], #False
#     'sample_batch': False,
#     "max_epoch": 600,
#     'start_epoch': 401,
#     'batch_size': 400,
#     'start_save': 180,
#     'conv': False,
#     'model_type': 'inv', #'dense', #'conv2', 
#     'data_type': 'atlas', #'atlas', #'simATAC', #'benchmark',
#     'lr': 1e-3, #dense 1e-3 conv 1e-2
#     'pos_w': 30,
#     'weight_decay': 5e-4,
#     'z_dim': 10,
#     'out_every': 2,
#     'ckpt_dir': './models/',
#     #simATAC
#     'setting': 2,
#     'signal': 0.35,
#     'frags': 3000,
#     'bin_size': 10000
# })

# solver = Trainer(args)

solver = Trainer(parser)
solver.warm_up()
solver.inv_train()
# solver.run_time()

# latent, labels, depth = solver.encode_adv(1000)
# result = latent.numpy()
# l = pd.DataFrame(labels, columns=['celltype'])
# d = depth.numpy()

# np.save(f'./results/latent/{args.name}_latent.npy', result)
# np.save(f'./results/latent/{args.name}_depth.npy', d)
# l.to_csv('./results/latent/{args.name}_labels.csv', index=False)
