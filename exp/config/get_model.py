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
from model.rung_parametric_gamma import RUNG_parametric_gamma
from model.rung_confidence_lambda import RUNG_confidence_lambda
from model.rung_percentile_gamma import RUNG_percentile_gamma
from model.rung_learnable_distance import RUNG_learnable_distance
from model.rung_learnable_combined import RUNG_learnable_combined
from model.rung_combined import RUNG_combined
from model.rung_combined_model import RUNG_combined_model


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
        # prefer cuda when available
        # PyTorch 1.12.1 with CUDA 11.3 supports compute capability 5.0+
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                print(
                    f"\nDetected GPU: {props.name} with compute capability "
                    f"{props.major}.{props.minor}"
                )
                # PyTorch 1.12.1 supports compute capability 5.0+
                if props.major < 5:
                    print(
                        f"GPU {props.name} with compute capability "
                        f"{props.major}.{props.minor} is not supported; falling back to CPU."
                    )
                    device = torch.device('cpu')
                else:
                    device = torch.device('cuda')
                    print(f"Using GPU: {props.name}\n")
            except Exception as e:
                print(f"Error querying GPU properties: {e}. Falling back to CPU.")
                device = torch.device('cpu')
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
    elif model_name == 'RUNG_parametric_gamma':
        # RUNG with 2-parameter exponential gamma decay schedule.
        # Replaces K separate log_lam parameters with:
        #   gamma^(k) = gamma_0 * decay_rate^k
        # This gives stronger gradient signals and more stable training.
        decay_rate_init     = custom_model_params.get('decay_rate_init', 0.85)
        scad_a              = custom_model_params.get('scad_a', 3.7)
        prop_step           = custom_model_params.get('prop_step', 10)
        dropout             = custom_model_params.get('dropout', 0.5)
        lam_hat             = custom_model_params.get('lam_hat', 0.9)
        model_pg_param = RUNG_parametric_gamma(
            in_dim=D,
            out_dim=C,
            hidden_dims=[64],
            lam_hat=lam_hat,
            gamma_0_init=gamma,        # gamma from --gamma CLI arg (same scale as RUNG_new_SCAD)
            decay_rate_init=decay_rate_init,
            scad_a=scad_a,
            prop_step=prop_step,
            dropout=dropout,
        ).to(device)
        return model_pg_param, custom_fit_params
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
    elif model_name == 'RUNG_percentile_gamma':
        # RUNG with percentile-based adaptive SCAD gamma.
        # Replaces the learned log_lam ParameterList with a data-driven
        # quantile computation — no gamma parameters, no gradient needed.
        scad_a             = custom_model_params.get('scad_a', 3.7)
        prop_step          = custom_model_params.get('prop_step', 10)
        dropout            = custom_model_params.get('dropout', 0.5)
        lam_hat            = custom_model_params.get('lam_hat', 0.9)
        percentile_q       = custom_model_params.get('percentile_q', 0.75)
        use_layerwise_q    = custom_model_params.get('use_layerwise_q', False)
        percentile_q_late  = custom_model_params.get('percentile_q_late', 0.65)
        model_pg = RUNG_percentile_gamma(
            in_dim            = D,
            out_dim           = C,
            hidden_dims       = [64],
            lam_hat           = lam_hat,
            percentile_q      = percentile_q,
            use_layerwise_q   = use_layerwise_q,
            percentile_q_late = percentile_q_late,
            scad_a            = scad_a,
            prop_step         = prop_step,
            dropout           = dropout,
        ).to(device)
        return model_pg, custom_fit_params
    elif model_name == 'RUNG_learnable_distance':
        # RUNG with configurable distance metric for edge suspiciousness.
        # Replaces hardcoded Euclidean distance with cosine, projection, or bilinear.
        # Extends RUNG_percentile_gamma with DistanceModule.
        scad_a             = custom_model_params.get('scad_a', 3.7)
        prop_step          = custom_model_params.get('prop_step', 10)
        dropout            = custom_model_params.get('dropout', 0.5)
        lam_hat            = custom_model_params.get('lam_hat', 0.9)
        percentile_q       = custom_model_params.get('percentile_q', 0.75)
        use_layerwise_q    = custom_model_params.get('use_layerwise_q', False)
        percentile_q_late  = custom_model_params.get('percentile_q_late', 0.65)
        distance_mode      = custom_model_params.get('distance_mode', 'cosine')
        proj_dim           = custom_model_params.get('proj_dim', 32)
        model_ld = RUNG_learnable_distance(
            in_dim            = D,
            out_dim           = C,
            hidden_dims       = [64],
            lam_hat           = lam_hat,
            percentile_q      = percentile_q,
            use_layerwise_q   = use_layerwise_q,
            percentile_q_late = percentile_q_late,
            distance_mode     = distance_mode,
            proj_dim          = proj_dim,
            scad_a            = scad_a,
            prop_step         = prop_step,
            dropout           = dropout,
        ).to(device)
        return model_ld, custom_fit_params
    elif model_name == 'RUNG_learnable_combined':
        # RUNG with cosine distance and learnable gamma constrained to (0,2).
        scad_a             = custom_model_params.get('scad_a', 3.7)
        prop_step          = custom_model_params.get('prop_step', 10)
        dropout            = custom_model_params.get('dropout', 0.5)
        lam_hat            = custom_model_params.get('lam_hat', 0.9)
        gamma_mode         = custom_model_params.get('gamma_mode', 'per_layer')
        model_lc = RUNG_learnable_combined(
            in_dim            = D,
            out_dim           = C,
            hidden_dims       = [64],
            lam_hat           = lam_hat,
            gamma_mode        = gamma_mode,
            scad_a            = scad_a,
            prop_step         = prop_step,
            dropout           = dropout,
        ).to(device)
        return model_lc, custom_fit_params
    elif model_name == 'RUNG_combined':
        # RUNG with cosine distance + percentile gamma — zero new parameters.
        # Combines the two strongest independent improvements:
        #   - Percentile gamma (from RUNG_percentile_gamma)
        #   - Cosine distance (from RUNG_learnable_distance mode='cosine')
        scad_a             = custom_model_params.get('scad_a', 3.7)
        prop_step          = custom_model_params.get('prop_step', 10)
        dropout            = custom_model_params.get('dropout', 0.5)
        lam_hat            = custom_model_params.get('lam_hat', 0.9)
        percentile_q       = custom_model_params.get('percentile_q', 0.75)
        use_layerwise_q    = custom_model_params.get('use_layerwise_q', False)
        percentile_q_late  = custom_model_params.get('percentile_q_late', 0.65)
        model_comb = RUNG_combined(
            in_dim            = D,
            out_dim           = C,
            hidden_dims       = [64],
            lam_hat           = lam_hat,
            percentile_q      = percentile_q,
            use_layerwise_q   = use_layerwise_q,
            percentile_q_late = percentile_q_late,
            scad_a            = scad_a,
            prop_step         = prop_step,
            dropout           = dropout,
        ).to(device)
        return model_comb, custom_fit_params
    elif model_name == 'RUNG_combined_model':
        # RUNG with all three mechanisms:
        # 1. Cosine distance (from RUNG_learnable_distance)
        # 2. Parametric gamma (from RUNG_parametric_gamma)  
        # 3. Percentile gamma (from RUNG_percentile_gamma)
        # Combined via learnable blend: gamma = alpha * gamma_param + (1-alpha) * gamma_data
        scad_a             = custom_model_params.get('scad_a', 3.7)
        prop_step          = custom_model_params.get('prop_step', 10)
        dropout            = custom_model_params.get('dropout', 0.5)
        lam_hat            = custom_model_params.get('lam_hat', 0.9)
        percentile_q       = custom_model_params.get('percentile_q', 0.75)
        decay_rate_init    = custom_model_params.get('decay_rate_init', 0.85)
        alpha_blend_init   = custom_model_params.get('alpha_blend_init', 0.5)
        model_cmb = RUNG_combined_model(
            in_dim            = D,
            out_dim           = C,
            hidden_dims       = [64],
            lam_hat           = lam_hat,
            percentile_q      = percentile_q,
            gamma_0_init      = gamma,
            decay_rate_init   = decay_rate_init,
            alpha_blend_init  = alpha_blend_init,
            scad_a            = scad_a,
            prop_step         = prop_step,
            dropout           = dropout,
        ).to(device)
        return model_cmb, custom_fit_params
    elif model_name == 'RUNG_percentile_adv':
        # RUNG_percentile_gamma trained with curriculum adversarial training.
        # Architecture is identical to RUNG_percentile_gamma.
        # Only the training procedure differs (uses AdversarialTrainer).
        scad_a             = custom_model_params.get('scad_a', 3.7)
        prop_step          = custom_model_params.get('prop_step', 10)
        dropout            = custom_model_params.get('dropout', 0.5)
        lam_hat            = custom_model_params.get('lam_hat', 0.9)
        percentile_q       = custom_model_params.get('percentile_q', 0.75)
        use_layerwise_q    = custom_model_params.get('use_layerwise_q', False)
        percentile_q_late  = custom_model_params.get('percentile_q_late', 0.65)
        model_pa = RUNG_percentile_gamma(
            in_dim            = D,
            out_dim           = C,
            hidden_dims       = [64],
            lam_hat           = lam_hat,
            percentile_q      = percentile_q,
            use_layerwise_q   = use_layerwise_q,
            percentile_q_late = percentile_q_late,
            scad_a            = scad_a,
            prop_step         = prop_step,
            dropout           = dropout,
        ).to(device)
        return model_pa, custom_fit_params
    elif model_name == 'RUNG_parametric_adv':
        # RUNG_parametric_gamma trained with curriculum adversarial training.
        # Architecture is identical to RUNG_parametric_gamma.
        # Only the training procedure differs (uses AdversarialTrainer).
        decay_rate_init     = custom_model_params.get('decay_rate_init', 0.85)
        scad_a              = custom_model_params.get('scad_a', 3.7)
        prop_step           = custom_model_params.get('prop_step', 10)
        dropout             = custom_model_params.get('dropout', 0.5)
        lam_hat             = custom_model_params.get('lam_hat', 0.9)
        model_paa = RUNG_parametric_gamma(
            in_dim=D,
            out_dim=C,
            hidden_dims=[64],
            lam_hat=lam_hat,
            gamma_0_init=gamma,
            decay_rate_init=decay_rate_init,
            scad_a=scad_a,
            prop_step=prop_step,
            dropout=dropout,
        ).to(device)
        return model_paa, custom_fit_params
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Valid choices: RUNG, RUNG_new, RUNG_new_SCAD, RUNG_new_L1, "
            f"RUNG_new_L2, RUNG_new_ADAPTIVE, RUNG_learnable_gamma, "
            f"RUNG_parametric_gamma, RUNG_confidence_lambda, RUNG_percentile_gamma, "
            f"RUNG_learnable_distance, RUNG_learnable_combined, RUNG_combined, RUNG_combined_model, "
            f"RUNG_percentile_adv, RUNG_parametric_adv, "
            f"GCN, GAT, APPNP, L1, MLP."
        )

