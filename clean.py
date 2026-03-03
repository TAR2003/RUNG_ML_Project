# append path
import sys

sys.path.append("./")

# models
from train_eval_data.fit import fit
from train_eval_data.fit_learnable_gamma import fit_learnable_gamma
from exp.config.get_model_cora import get_model_default_cora
from exp.config.get_model_citeseer import get_model_default_citeseer
from exp.config.get_model import get_model_default
# dataset, train and eval
from train_eval_data.get_dataset import get_dataset, get_splits
from utils import accuracy
import copy
import yaml
from exp.result_io import save_acc, rep_save_model

# computation pkgs
import torch
from torch import nn
import numpy as np
import time
time_str = time.strftime('%Y-%m-%d-%H-%M')
import argparse

import os 

path = "./"

parser = argparse.ArgumentParser(description='Train classification network')

# model setting
parser.add_argument('--model',type=str, default='RUNG')
parser.add_argument('--norm',type=str, default='MCP')
parser.add_argument('--penalty',type=str, default=None,
                    choices=['mcp', 'scad', 'l1', 'l2', 'adaptive', None],
                    help='Penalty function override (maps to --norm). '
                         'mcp=MCP, scad=SCAD, l1=L1, l2=L2, adaptive=ADAPTIVE. '
                         'If set, overrides --norm for RUNG models.')
parser.add_argument('--gamma',type=float, default=6.0)
parser.add_argument('--data',type=str, default='cora')

# RUNG_learnable_gamma specific arguments
parser.add_argument('--gamma_init_strategy', type=str, default='uniform',
                    choices=['uniform', 'decreasing', 'increasing'],
                    help='How to initialise gamma across layers for RUNG_learnable_gamma. '
                         'decreasing is theoretically motivated by shrinking feature '
                         'differences with depth.')
parser.add_argument('--gamma_lr_factor', type=float, default=0.2,
                    help='LR multiplier for gamma parameters in RUNG_learnable_gamma. '
                         'gamma_lr = lr * gamma_lr_factor. Recommended range: 0.05–0.5.')
parser.add_argument('--gamma_reg_strength', type=float, default=0.0,
                    help='Regularisation strength for gamma params in RUNG_learnable_gamma. '
                         '0 = disabled. Try 0.01 if gammas diverge during training.')

# fitting setting
parser.add_argument('--lr',type=float, default=5e-2)
parser.add_argument('--weight_decay',type=float, default=5e-4)
parser.add_argument('--max_epoch',type=int, default=300)

args = parser.parse_args()
# Compound model names encode both model and norm (e.g. RUNG_new_SCAD).
# Normalise them into separate args.model / args.norm before anything else.
_COMPOUND_MODEL_MAP = {
    'RUNG_new_SCAD':         ('RUNG_new', 'SCAD'),
    'RUNG_new_L1':           ('RUNG_new', 'L1'),
    'RUNG_new_L2':           ('RUNG_new', 'L2'),
    'RUNG_new_ADAPTIVE':     ('RUNG_new', 'ADAPTIVE'),
    'RUNG_SCAD':             ('RUNG',     'SCAD'),
    'RUNG_L1':               ('RUNG',     'L1'),
    'RUNG_L2':               ('RUNG',     'L2'),
    # RUNG_learnable_gamma is NOT mapped here — it is handled as a standalone
    # model name in get_model_default and in the training dispatch below.
}
if args.model in _COMPOUND_MODEL_MAP:
    args.model, args.norm = _COMPOUND_MODEL_MAP[args.model]

if args.model == 'APPNP':
    args.norm = 'L2'
elif args.model == 'L1':
    args.norm = 'L1'
# --penalty overrides --norm (convenience alias with lower-case names)
_PENALTY_TO_NORM = {'mcp': 'MCP', 'scad': 'SCAD', 'l1': 'L1', 'l2': 'L2', 'adaptive': 'ADAPTIVE'}
if args.penalty is not None:
    args.norm = _PENALTY_TO_NORM[args.penalty.lower()]


def clean_rep(model, train_param, dataset_name, seed=None):
    A, X, y = get_dataset(dataset_name)
    sp = get_splits(y)

    device = next(model.parameters()).device
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)

    acc, models = [], []
    for train_idx, val_idx, test_idx in sp:
        train_idx = train_idx.to(device)
        val_idx   = val_idx.to(device)
        test_idx  = test_idx.to(device)
        cur_model = copy.deepcopy(model)
        torch.manual_seed(seed if seed is not None else 0)
        if args.model in  ['GCN','GAT']:
            cur_model.fit((A, X), y, train_idx, val_idx, progress=False, **train_param)
        elif args.model in ('RUNG', 'RUNG_new', 'MLP', 'L1', 'APPNP'):
            fit(cur_model, A, X, y, train_idx, val_idx, **train_param)
        elif args.model == 'RUNG_learnable_gamma':
            fit_learnable_gamma(
                cur_model, A, X, y, train_idx, val_idx,
                gamma_lr_factor=args.gamma_lr_factor,
                gamma_reg_strength=args.gamma_reg_strength,
                **train_param,
            )
        
        cur_model.eval()
        acc.append(accuracy(cur_model(A, X)[test_idx, :], y[test_idx]).cpu().item())
        print("Acc:",acc)
        models.append(cur_model)
    return acc, models


def make_clean_model_and_save(do_save_model=False, do_save_acc=False, rep_num=5, model_name_arg=None):
    clean_result_fname = path + f"exp/result/{args.data}/clean_{args.model}_{args.gamma}.yaml"
    
    # get model name
    
    model_config = {'gamma': args.gamma, 'norm': args.norm}
    # Pass RUNG_learnable_gamma-specific params into model config so
    # get_model_default can forward them to the constructor.
    if args.model == 'RUNG_learnable_gamma':
        model_config['gamma_init_strategy'] = args.gamma_init_strategy

    model_ls = [
        [args.model, model_config, {'lr':args.lr, 'weight_decay':args.weight_decay,'max_epoch': args.max_epoch}],
    ]
    

    
    for model_name, model_config, fit_config in model_ls if model_name_arg is None else model_name_arg:
        acc, models = [], []
        for seed in range(rep_num):
            a, m = clean_rep(
                *get_model(
                    args.data,
                    model_name, 
                    custom_model_params=model_config, 
                    custom_fit_params=fit_config, 
                    seed=seed
                ), 
                args.data, 
                seed=seed, 
            )
            acc += a
            models.append(m)
        
        models = [m[i] for i in range(len(models[0])) for m in models]

        print(f'model {model_name} done, clean acc: {np.mean(acc)}±{np.std(acc)}')

        if do_save_acc:
            save_acc(acc, acc, clean_result_fname, model_name=model_name)
        if do_save_model:
            rep_save_model(f"{model_name}_{args.norm}_{args.gamma}", 0, models, dataset_name=args.data)
        




if __name__ == '__main__':

    os.makedirs(path+f'log/{args.data}/clean', exist_ok=True)
    sys.stdout = open(path+f'log/{args.data}/clean/{args.model}_{args.norm}_{args.gamma}.log', 'w', buffering=1)
    

    get_model = get_model_default
    make_clean_model_and_save(do_save_acc=True, do_save_model=True, rep_num=1, model_name_arg=None)
    
    sys.stdout.close()



