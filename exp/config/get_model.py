# dataset, train and eval
from train_eval_data.get_dataset import get_dataset, get_splits
from utils import accuracy
import copy
import yaml
from exp.result_io import save_acc, rep_save_model
from typeguard import typechecked

# models
# from model.softmedian import SoftMedianAPPNP, SoftMedianPropagation
from model.mlp import MLP
from model.att_func import get_l12_att_func, get_default_att_func, get_log_att_func, get_mask_att_func, get_step_p_norm_att_func, get_soft_step_l21_att_func, get_mcp_att_func, get_scad_att_func
from model.penalty import PenaltyFunction
from model.rung import RUNG
from model.rung_learnable_gamma import RUNG_learnable_gamma
from model.rung_confidence_lambda import RUNG_confidence_lambda


# preprocessing for sontructing models
# from model.plumbing import GraphSequential, PreprocessA, PreprocessX
from collections import OrderedDict

# computation pkgs
import torch
from torch import nn
import numpy as np

import re
import gb

def get_model_default(
    dataset, model_name, custom_model_params={}, custom_fit_params={}, as_paper=True, seed=None, D=None, device=None
):
    if device is None:
        # prefer cuda when available, but validate compute capability against
        # the PyTorch build.  Many older GPUs (e.g. GeForce MX130, capability
        # 5.0) are no longer supported by recent PyTorch binaries and will
        # produce ``no kernel image is available for execution on the device``
        # errors observed on the local machine.  In those cases we force CPU
        # execution instead of crashing.
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                # PyTorch wheels currently support sm_70 and higher; adjust
                # threshold if the build changes.
                if props.major < 7:
                    print(
                        f"Detected GPU {props.name} with compute capability"
                        f" {props.major}.{props.minor} which is unsupported by "
                        "this PyTorch installation; falling back to CPU."
                    )
                    device = torch.device('cpu')
                else:
                    device = torch.device('cuda')
            except Exception:
                # if for some reason we can't query properties, still try
                # cuda and let the normal error handling occur.
                device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    torch.manual_seed(0 if seed is None else seed)

    A, X, y = get_dataset(dataset)
    sp = get_splits(y)
    train_idx, val_idx, test_idx = sp[0]

    D = X.shape[1] if D is None else D
    C = y.unique().shape[0]

    # Resolve norm / penalty into a w_func callable + optional penalty flag for RUNG
    norm  = custom_model_params.get('norm', 'MCP')
    gamma = custom_model_params.get('gamma', 3.0)

    def _build_rung(w_func, penalty_flag=None):
        return RUNG(
            D, C, [64], w_func, 0.9,
            penalty=penalty_flag,
            gamma=gamma,
        ).to(device)

    if model_name == 'RUNG':
        norm_upper = norm.upper()
        if norm_upper == 'MCP':
            return _build_rung(get_mcp_att_func(gamma)), custom_fit_params
        elif norm_upper == 'SCAD':
            # SCAD: use lam = gamma/3.7 so that the cutoff a*lam = gamma
            # matches the MCP cutoff for fair comparison.
            lam_scad = gamma / 3.7
            return _build_rung(get_scad_att_func(lam_scad, 3.7)), custom_fit_params
        elif norm_upper == 'L1':
            return _build_rung(get_l12_att_func('L1')), custom_fit_params
        elif norm_upper == 'L2':
            return _build_rung(get_l12_att_func('L2')), custom_fit_params
        elif norm_upper == 'ADAPTIVE':
            # Adaptive mode: w_func is a dummy (MCP); the actual weights
            # are computed in RUNG.forward when penalty='adaptive'.
            return _build_rung(get_mcp_att_func(gamma), penalty_flag='adaptive'), custom_fit_params
        else:
            raise ValueError(
                f"Unknown norm '{norm}' for RUNG. "
                f"Choose from: MCP, SCAD, L1, L2, ADAPTIVE."
            )
    elif model_name == 'RUNG_new':
        # RUNG_new — uses the refactored model/penalty.py code path (PenaltyFunction class).
        # Numerical differences from baseline RUNG: eps=1e-8 (vs ep=0.01 offset in att_func.py),
        # and SCAD / ADAPTIVE penalties are only cleanly supported via this path.
        norm_upper = norm.upper()
        if norm_upper == 'MCP':
            w_func = PenaltyFunction.get_w_func('mcp', gamma)
            return _build_rung(w_func), custom_fit_params
        elif norm_upper == 'SCAD':
            # lam = gamma/3.7 so zero-region a*lam = gamma matches MCP cutoff
            w_func = PenaltyFunction.get_w_func('scad', gamma / 3.7)
            return _build_rung(w_func), custom_fit_params
        elif norm_upper == 'L1':
            w_func = PenaltyFunction.get_w_func('l1', gamma)
            return _build_rung(w_func), custom_fit_params
        elif norm_upper == 'L2':
            w_func = PenaltyFunction.get_w_func('l2', gamma)
            return _build_rung(w_func), custom_fit_params
        elif norm_upper == 'ADAPTIVE':
            # MCP as base w_func; adaptive weighting computed inside forward()
            w_func = PenaltyFunction.get_w_func('mcp', gamma)
            return _build_rung(w_func, penalty_flag='adaptive'), custom_fit_params
        else:
            raise ValueError(
                f"Unknown norm '{norm}' for RUNG_new. "
                f"Choose from: MCP, SCAD, L1, L2, ADAPTIVE."
            )
    elif model_name == 'GCN':
        return gb.model.GCN(n_feat=D, n_class=C, hidden_dims=[64], dropout=0.5).to(device), custom_fit_params
    elif model_name == 'GAT':
        return gb.model.GAT(num_of_layers=2, num_heads_per_layer=[4,1], num_features_per_layer=[D,16,C], add_skip_connection=True, bias=True,
                 dropout=0.1, log_attention_weights=False).to(device), custom_fit_params
    elif model_name in ['APPNP', 'L1']:
        return RUNG(D, C, [64], get_l12_att_func(custom_model_params['norm']), 0.9).to(device), custom_fit_params
    elif model_name == 'MLP':
        return RUNG(D, C, [64], get_mcp_att_func(gamma), 0.9, prop_step=0).to(device), custom_fit_params
    # Compound aliases — resolve to the canonical model + norm branch above
    elif model_name == 'RUNG_new_SCAD':
        w_func = PenaltyFunction.get_w_func('scad', gamma / 3.7)
        return _build_rung(w_func), custom_fit_params
    elif model_name == 'RUNG_new_L1':
        w_func = PenaltyFunction.get_w_func('l1', gamma)
        return _build_rung(w_func), custom_fit_params
    elif model_name == 'RUNG_new_L2':
        w_func = PenaltyFunction.get_w_func('l2', gamma)
        return _build_rung(w_func), custom_fit_params
    elif model_name == 'RUNG_new_ADAPTIVE':
        w_func = PenaltyFunction.get_w_func('mcp', gamma)
        return _build_rung(w_func, penalty_flag='adaptive'), custom_fit_params
    elif model_name == 'RUNG_learnable_gamma':
        # Extract extra kwargs specific to RUNG_learnable_gamma; fall back to
        # sensible defaults that mirror RUNG_new_SCAD where applicable.
        gamma_init_strategy = custom_model_params.get('gamma_init_strategy', 'uniform')
        scad_a              = custom_model_params.get('scad_a', 3.7)
        prop_step           = custom_model_params.get('prop_step', 10)
        dropout             = custom_model_params.get('dropout', 0.5)
        lam_hat             = custom_model_params.get('lam_hat', 0.9)
        model_lg = RUNG_learnable_gamma(
            in_dim=D,
            out_dim=C,
            hidden_dims=[64],
            lam_hat=lam_hat,
            gamma_init=gamma,          # gamma from --gamma CLI arg (same scale as RUNG_new_SCAD)
            gamma_init_strategy=gamma_init_strategy,
            scad_a=scad_a,
            prop_step=prop_step,
            dropout=dropout,
        ).to(device)
        return model_lg, custom_fit_params
    elif model_name == 'RUNG_confidence_lambda':
        # RUNG with per-layer learnable SCAD gamma AND per-node confidence-weighted lambda.
        # Extends RUNG_learnable_gamma with one new parameter: raw_alpha (sharpness).
        gamma_init_strategy = custom_model_params.get('gamma_init_strategy', 'uniform')
        scad_a              = custom_model_params.get('scad_a', 3.7)
        prop_step           = custom_model_params.get('prop_step', 10)
        dropout             = custom_model_params.get('dropout', 0.5)
        lam_hat             = custom_model_params.get('lam_hat', 0.9)
        alpha_init          = custom_model_params.get('alpha_init', 1.0)
        confidence_mode     = custom_model_params.get('confidence_mode', 'protect_uncertain')
        normalize_lambda    = custom_model_params.get('normalize_lambda', True)
        model_cl = RUNG_confidence_lambda(
            in_dim=D,
            out_dim=C,
            hidden_dims=[64],
            lam_hat=lam_hat,
            gamma_init=gamma,           # gamma from --gamma CLI arg
            gamma_init_strategy=gamma_init_strategy,
            scad_a=scad_a,
            prop_step=prop_step,
            dropout=dropout,
            alpha_init=alpha_init,
            confidence_mode=confidence_mode,
            normalize_lambda=normalize_lambda,
        ).to(device)
        return model_cl, custom_fit_params
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Valid choices: RUNG, RUNG_new, RUNG_new_SCAD, RUNG_new_L1, "
            f"RUNG_new_L2, RUNG_new_ADAPTIVE, RUNG_learnable_gamma, "
            f"RUNG_confidence_lambda, GCN, GAT, APPNP, L1, MLP."
        )

