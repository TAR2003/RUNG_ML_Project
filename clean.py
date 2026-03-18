# append path
import sys

sys.path.append("./")

# models
from train_eval_data.fit import fit
from train_eval_data.fit_learnable_gamma import fit_learnable_gamma
from train_eval_data.fit_parametric_gamma import fit_parametric_gamma
from train_eval_data.fit_combined_model import fit_combined_model
from train_eval_data.fit_confidence_lambda import fit_confidence_lambda
from train_eval_data.fit_percentile_gamma import fit_percentile_gamma
from train_eval_data.fit_learnable_distance import fit_learnable_distance
from train_eval_data.fit_learnable_combined import fit_learnable_combined
from train_eval_data.fit_percentile_adv import fit_percentile_adv
from train_eval_data.fit_percentile_adv_v2 import fit_percentile_adv_v2
from train_eval_data.fit_parametric_adv import fit_parametric_adv
from experiments.run_ablation import pgd_attack
from exp.config.get_model_cora import get_model_default_cora
from exp.config.get_model_citeseer import get_model_default_citeseer
from exp.config.get_model import get_model_default
# dataset, train and eval
from train_eval_data.get_dataset import get_dataset, get_splits
from utils import accuracy, get_log_identifier
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
parser.add_argument('--gamma_lr_factor', type=float, default=0.3,
                    help='LR multiplier for gamma parameters in RUNG_learnable_gamma. '
                         'gamma_lr = lr * gamma_lr_factor. Recommended range: 0.05–0.5.')
parser.add_argument('--gamma_reg_strength', type=float, default=0.0,
                    help='Regularisation strength for gamma params in RUNG_learnable_gamma. '
                         '0 = disabled. Try 0.01 if gammas diverge during training.')

# RUNG_parametric_gamma specific arguments
parser.add_argument('--decay_rate_init', type=float, default=0.85,
                    help='Initial gamma decay rate per layer for RUNG_parametric_gamma. '
                         '0.85 means gamma shrinks by 15%% per layer. Range (0,1). '
                         'Typical search: 0.70, 0.80, 0.85, 0.90, 0.95.')
parser.add_argument('--decay_rate_reg_strength', type=float, default=0.0,
                    help='Regularisation strength for decay_rate in RUNG_parametric_gamma. '
                         '0 = disabled. Try 0.01 if decay_rate diverges during training.')

# RUNG_confidence_lambda specific arguments
parser.add_argument('--alpha_init', type=float, default=1.0,
                    help='Initial sharpness for confidence-to-lambda mapping. '
                         'alpha=1 is linear. alpha>1 amplifies confidence differences. '
                         'alpha is learnable — this is just the initialisation.')
parser.add_argument('--alpha_lr_factor', type=float, default=0.1,
                    help='LR multiplier for alpha param in RUNG_confidence_lambda. '
                         'alpha_lr = lr * alpha_lr_factor. Recommended: 0.05–0.2.')
parser.add_argument('--alpha_reg_strength', type=float, default=0.001,
                    help='Alpha regularisation strength for RUNG_confidence_lambda. '
                         '0 = disabled. Penalises alpha deviating from 1.0.')
parser.add_argument('--confidence_mode', type=str, default='protect_uncertain',
                    choices=['protect_uncertain', 'protect_confident', 'symmetric'],
                    help="How to map confidence to lambda. "
                         "'protect_uncertain': uncertain nodes get higher lambda. "
                         "'protect_confident': confident nodes get higher lambda. "
                         "'symmetric': mid-confidence nodes get highest lambda.")
parser.add_argument('--normalize_lambda', type=lambda x: x.lower() != 'false',
                    default=True,
                    help='If True, normalise per-node lambdas so mean = lambda_base. '
                         'Recommended True for fair comparison with RUNG_learnable_gamma.')
parser.add_argument('--warmup_epochs', type=int, default=50,
                    help='Epochs to train only MLP before unfreezing gamma + alpha. '
                         'Ensures MLP confidences are meaningful before calibration. '
                         'Set to 0 to disable warmup.')

# RUNG_percentile_gamma specific arguments
parser.add_argument('--percentile_q', type=float, default=0.75,
                    help='Percentile for gamma computation in RUNG_percentile_gamma. '
                         'gamma^(k) = quantile(y_edges^(k), percentile_q). '
                         'Range (0, 1). Higher = lighter pruning. '
                         'Recommended search: 0.50, 0.60, 0.75, 0.85, 0.90, 0.95.')
parser.add_argument('--use_layerwise_q', type=lambda x: x.lower() != 'false',
                    default=False,
                    help='If True, use different percentile_q for early vs late layers. '
                         'Early layers use --percentile_q; late layers use --percentile_q_late.')
parser.add_argument('--percentile_q_late', type=float, default=0.65,
                    help='Percentile q for late layers when use_layerwise_q=True. '
                         'Should be <= percentile_q for more aggressive late-layer pruning.')

# RUNG_learnable_distance specific arguments
parser.add_argument('--distance_mode', type=str, default='cosine',
                    choices=['cosine', 'projection', 'bilinear'],
                    help='Distance metric for edge suspiciousness in RUNG_learnable_distance. '
                         'cosine: 0 parameters, scale-invariant (start here). '
                         'projection: learnable MLP projection. '
                         'bilinear: learnable linear projection.')
parser.add_argument('--proj_dim', type=int, default=32,
                    help='Projection dimension for projection/bilinear distance modes. '
                         'Ignored for cosine mode.')
parser.add_argument('--dist_lr_factor', type=float, default=0.5,
                    help='LR multiplier for distance module parameters in RUNG_learnable_distance. '
                         'Only used if distance_mode is projection or bilinear. '
                         'Default 0.5 = distance LR is half of base LR.')
parser.add_argument('--gamma_mode', type=str, default='per_layer',
                choices=['per_layer', 'schedule'],
                help='Learnable gamma mode for RUNG_learnable_combined. '
                    'per_layer = one gamma per layer, schedule = 2-parameter decay.')

# RUNG_combined_model specific arguments
parser.add_argument('--alpha_blend_init', type=float, default=0.5,
                    help='Initial blend weight for parametric vs percentile gamma in RUNG_combined_model. '
                         '0.0 = pure percentile (data-driven), '
                         '1.0 = pure parametric (learned schedule), '
                         '0.5 = equal blend (recommended starting point). '
                         'Learnable during training.')

# fitting setting
parser.add_argument('--lr',type=float, default=5e-2)
parser.add_argument('--weight_decay',type=float, default=5e-4)
parser.add_argument('--max_epoch',type=int, default=300)

# Adversarial training arguments (for RUNG_percentile_adv, RUNG_parametric_adv)
parser.add_argument('--adv_alpha', type=float, default=0.7,
                    help='Weight on clean loss in adversarial training. '
                         'L = adv_alpha * L_clean + (1-adv_alpha) * L_adv. '
                         '0.7 = 70%% clean, 30%% adversarial (recommended). '
                         'Increase toward 1.0 if clean accuracy drops. ')
parser.add_argument('--attack_freq', type=int, default=5,
                    help='Regenerate adversarial graph every N epochs. '
                         'Higher = faster. 5 is balanced (recommended).')
parser.add_argument('--train_pgd_steps', type=int, default=20,
                    help='PGD iterations during adversarial training. '
                         '20-50 is sufficient. Fewer = faster training.')
parser.add_argument('--curriculum_budgets', type=float, nargs='+', default=None,
                    help='Attack budgets per curriculum phase. '
                         'Example: 0.05 0.10 0.20 0.40')
parser.add_argument('--curriculum_epochs', type=int, nargs='+', default=None,
                    help='Epoch counts per curriculum phase. '
                         'Example: 50 50 100  (last is ignored, stays forever)')

# Adversarial training V2 arguments (for RUNG_percentile_adv_v2)
# These override the V1 defaults for the fixed version
parser.add_argument('--adv_alpha_v2', type=float, default=0.85,
                    help='[V2] Weight on clean loss. Default 0.85 (stronger base model focus).')
parser.add_argument('--train_pgd_steps_v2', type=int, default=100,
                    help='[V2] PGD steps during training. Default 100 (matches test strength).')
parser.add_argument('--attack_freq_v2', type=int, default=3,
                    help='[V2] Regenerate attack every N epochs. Default 3 (fresher examples).')
parser.add_argument('--warmup_epochs_v2', type=int, default=100,
                    help='[V2] Clean-only training epochs. Default 100 (MLP stabilization).')

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
        elif args.model == 'RUNG_parametric_gamma':
            fit_parametric_gamma(
                cur_model, A, X, y, train_idx, val_idx,
                gamma_lr_factor=args.gamma_lr_factor,
                gamma_reg_strength=args.gamma_reg_strength,
                **train_param,
            )
        elif args.model == 'RUNG_confidence_lambda':
            fit_confidence_lambda(
                cur_model, A, X, y, train_idx, val_idx,
                gamma_lr_factor=args.gamma_lr_factor,
                alpha_lr_factor=args.alpha_lr_factor,
                gamma_reg_strength=args.gamma_reg_strength,
                alpha_reg_strength=args.alpha_reg_strength,
                warmup_epochs=args.warmup_epochs,
                **train_param,
            )
        elif args.model == 'RUNG_percentile_gamma':
            fit_percentile_gamma(
                cur_model, A, X, y, train_idx, val_idx,
                **train_param,
            )
        elif args.model == 'RUNG_learnable_distance':
            fit_learnable_distance(
                cur_model, A, X, y, train_idx, val_idx,
                dist_lr_factor=args.dist_lr_factor,
                **train_param,
            )
        elif args.model == 'RUNG_learnable_combined':
            fit_learnable_combined(
                cur_model, A, X, y, train_idx, val_idx,
                gamma_lr_factor=args.gamma_lr_factor,
                **train_param,
            )
        elif args.model == 'RUNG_combined_model':
            fit_combined_model(
                cur_model, A, X, y, train_idx, val_idx,
                gamma_lr_factor=args.gamma_lr_factor,
                gamma_reg_strength=args.gamma_reg_strength,
                **train_param,
            )
        elif args.model == 'RUNG_percentile_adv':
            fit_percentile_adv(
                cur_model, A, X, y, train_idx, val_idx, test_idx,
                attack_fn=pgd_attack,
                alpha=args.adv_alpha,
                attack_freq=args.attack_freq,
                train_pgd_steps=args.train_pgd_steps,
                curriculum_budgets=args.curriculum_budgets,
                curriculum_epochs=args.curriculum_epochs,
                **train_param,
            )
        elif args.model == 'RUNG_percentile_adv_v2':
            # Fixed adversarial training with stronger defaults
            fit_percentile_adv_v2(
                cur_model, A, X, y, train_idx, val_idx, test_idx,
                attack_fn=pgd_attack,
                alpha=args.adv_alpha_v2,
                train_pgd_steps=args.train_pgd_steps_v2,
                attack_freq=args.attack_freq_v2,
                warmup_epochs=args.warmup_epochs_v2,
                max_epoch=args.max_epoch,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
            )
        elif args.model == 'RUNG_parametric_adv':
            fit_parametric_adv(
                cur_model, A, X, y, train_idx, val_idx, test_idx,
                attack_fn=pgd_attack,
                alpha=args.adv_alpha,
                attack_freq=args.attack_freq,
                train_pgd_steps=args.train_pgd_steps,
                curriculum_budgets=args.curriculum_budgets,
                curriculum_epochs=args.curriculum_epochs,
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
    # Pass RUNG_parametric_gamma-specific params into model config.
    elif args.model == 'RUNG_parametric_gamma':
        model_config['decay_rate_init'] = args.decay_rate_init
    # Pass RUNG_confidence_lambda-specific params into model config.
    elif args.model == 'RUNG_confidence_lambda':
        model_config['gamma_init_strategy'] = args.gamma_init_strategy
        model_config['alpha_init']          = args.alpha_init
        model_config['confidence_mode']     = args.confidence_mode
        model_config['normalize_lambda']    = args.normalize_lambda
    # Pass RUNG_percentile_gamma-specific params into model config.
    elif args.model == 'RUNG_percentile_gamma':
        model_config['percentile_q']      = args.percentile_q
        model_config['use_layerwise_q']   = args.use_layerwise_q
        model_config['percentile_q_late'] = args.percentile_q_late
    # Pass RUNG_learnable_distance-specific params into model config.
    elif args.model == 'RUNG_learnable_distance':
        model_config['percentile_q']      = args.percentile_q
        model_config['use_layerwise_q']   = args.use_layerwise_q
        model_config['percentile_q_late'] = args.percentile_q_late
        model_config['distance_mode']     = args.distance_mode
        model_config['proj_dim']          = args.proj_dim
    elif args.model == 'RUNG_learnable_combined':
        model_config['gamma_mode'] = args.gamma_mode
    # Pass RUNG_combined_model-specific params into model config.
    elif args.model == 'RUNG_combined_model':
        model_config['percentile_q']      = args.percentile_q
        model_config['decay_rate_init']   = args.decay_rate_init
        model_config['alpha_blend_init']  = args.alpha_blend_init
    # RUNG_percentile_adv uses RUNG_percentile_gamma architecture
    elif args.model == 'RUNG_percentile_adv':
        model_config['percentile_q']      = args.percentile_q
        model_config['use_layerwise_q']   = args.use_layerwise_q
        model_config['percentile_q_late'] = args.percentile_q_late
    # RUNG_percentile_adv_v2 uses RUNG_percentile_gamma architecture (fixed version)
    elif args.model == 'RUNG_percentile_adv_v2':
        model_config['percentile_q']      = args.percentile_q
        model_config['use_layerwise_q']   = args.use_layerwise_q
        model_config['percentile_q_late'] = args.percentile_q_late
    # RUNG_parametric_adv uses RUNG_parametric_gamma architecture
    elif args.model == 'RUNG_parametric_adv':
        model_config['decay_rate_init']   = args.decay_rate_init

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
    
    # Generate model-specific log identifier (uses actual hyperparameters, not defaults)
    log_identifier = get_log_identifier(args.model, args)
    sys.stdout = open(path+f'log/{args.data}/clean/{log_identifier}.log', 'w', buffering=1)
    

    get_model = get_model_default
    make_clean_model_and_save(do_save_acc=True, do_save_model=True, rep_num=1, model_name_arg=None)
    
    sys.stdout.close()



