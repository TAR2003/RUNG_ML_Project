"""
utils package.

Re-exports all symbols from the root utils.py so that existing imports
  ``from utils import add_loops, ...``
continue to work after utils.py is shadowed by this package.

Also exposes the new metrics sub-module:
  ``from utils.metrics import compute_estimation_bias, ...``
"""

import torch

# ============================================================================
# Model-specific logging identifier
# ============================================================================

def get_log_identifier(model_name: str, args) -> str:
    """
    Generate a model-specific log identifier that reflects the actual hyperparameters
    being used for training/attack, not generic defaults.
    
    Returns a string used in log filenames like:
      RUNG_normMCP_gamma6.0
      RUNG_percentile_gamma_q0.75
      RUNG_learnable_distance_q0.75_dist_cosine
      RUNG_combined_model_q0.75_decay0.85_alpha0.5
      etc.
    
    This ensures logs and figures correctly reflect each model's actual configuration.
    """
    
    if model_name in ('RUNG', 'RUNG_new', 'MLP', 'L1', 'APPNP', 'GCN', 'GAT'):
        # Standard penalty-based models use norm and gamma
        norm = getattr(args, 'norm', 'MCP')
        gamma = getattr(args, 'gamma', 6.0)
        return f"{model_name}_norm{norm}_gamma{gamma}"
    
    elif model_name == 'RUNG_learnable_gamma':
        # Learnable gamma model: uses gamma_lr_factor, gamma_reg_strength, init strategy
        init_strat = getattr(args, 'gamma_init_strategy', 'uniform')
        reg = getattr(args, 'gamma_reg_strength', 0.0)
        return f"{model_name}_init{init_strat}_reg{reg}"
    
    elif model_name == 'RUNG_parametric_gamma':
        # Parametric gamma: uses decay_rate
        decay_rate = getattr(args, 'decay_rate_init', 0.85)
        reg = getattr(args, 'decay_rate_reg_strength', 0.0)
        return f"{model_name}_decay{decay_rate}_reg{reg}"
    
    elif model_name == 'RUNG_percentile_gamma':
        # Percentile-based gamma: uses percentile_q and optional layerwise variant
        q = getattr(args, 'percentile_q', 0.75)
        use_lw = getattr(args, 'use_layerwise_q', False)
        q_late = getattr(args, 'percentile_q_late', 0.65)
        
        if use_lw:
            return f"{model_name}_q{q}_qLate{q_late}"
        else:
            return f"{model_name}_q{q}"
    
    elif model_name == 'RUNG_learnable_distance':
        # Learnable distance: uses percentile_q, distance_mode, proj_dim
        q = getattr(args, 'percentile_q', 0.75)
        dist_mode = getattr(args, 'distance_mode', 'cosine')
        use_lw = getattr(args, 'use_layerwise_q', False)
        q_late = getattr(args, 'percentile_q_late', 0.65)
        
        if use_lw:
            q_suffix = f"_q{q}_qLate{q_late}"
        else:
            q_suffix = f"_q{q}"
        
        if dist_mode == 'cosine':
            return f"{model_name}_dist{dist_mode}{q_suffix}"
        else:
            proj = getattr(args, 'proj_dim', 32)
            return f"{model_name}_dist{dist_mode}_proj{proj}{q_suffix}"
    
    elif model_name == 'RUNG_learnable_combined':
        # Learnable combined: uses gamma_mode
        gamma_mode = getattr(args, 'gamma_mode', 'per_layer')
        return f"{model_name}_mode{gamma_mode}"
    
    elif model_name == 'RUNG_combined':
        # RUNG_combined: percentile + cosine distance
        q = getattr(args, 'percentile_q', 0.75)
        use_lw = getattr(args, 'use_layerwise_q', False)
        q_late = getattr(args, 'percentile_q_late', 0.65)
        
        if use_lw:
            return f"{model_name}_q{q}_qLate{q_late}"
        else:
            return f"{model_name}_q{q}"
    
    elif model_name == 'RUNG_combined_model':
        # Combined model: percentile + parametric + cosine
        q = getattr(args, 'percentile_q', 0.75)
        decay_rate = getattr(args, 'decay_rate_init', 0.85)
        alpha = getattr(args, 'alpha_blend_init', 0.5)
        
        return f"{model_name}_q{q}_decay{decay_rate}_a{alpha}"
    
    elif model_name == 'RUNG_confidence_lambda':
        # Confidence-lambda: uses confidence_mode and alpha_init
        conf_mode = getattr(args, 'confidence_mode', 'protect_uncertain')
        alpha = getattr(args, 'alpha_init', 1.0)
        
        return f"{model_name}_{conf_mode[:4]}_a{alpha}"
    
    elif model_name in ('RUNG_percentile_adv', 'RUNG_percentile_adv_v2', 'RUNG_parametric_adv'):
        # Adversarial training models
        q = getattr(args, 'percentile_q', 0.75)
        adv_alpha = getattr(args, 'adv_alpha', 0.7) if model_name == 'RUNG_percentile_adv' else getattr(args, 'adv_alpha_v2', 0.85)
        return f"{model_name}_q{q}_advAlpha{adv_alpha}"
    
    else:
        # Fallback for unknown models
        norm = getattr(args, 'norm', 'unknown')
        gamma = getattr(args, 'gamma', 'unknown')
        return f"{model_name}_norm{norm}_gamma{gamma}"

# ---------------------------------------------------------------------------
# Re-export everything from the root utils.py (graph helpers, metrics, etc.)
# ---------------------------------------------------------------------------

# Graph helpers
def add_loops(A):
    n = A.shape[-1]
    return A + torch.eye(n, device=A.device)

def sym_norm(A):
    Dsq = A.sum(-1).sqrt()
    return A / Dsq / Dsq.unsqueeze(-1)

# Classification metric
def accuracy(scores, y_true):
    return (scores.argmax(dim=-1) == y_true).count_nonzero(dim=-1) / y_true.shape[0]

# Python helper
def sub_dict(dct, *filter_keys, optional=False):
    if not optional:
        return {key: dct[key] for key in filter_keys}
    else:
        return {key: dct[key] for key in dct if key in filter_keys}

# Sparse tensor helper
def sp_new_values(t, values):
    out = torch.sparse_coo_tensor(t._indices(), values, t.shape)
    if t.is_coalesced():
        with torch.no_grad():
            out._coalesced_(True)
    return out

# Pairwise squared Euclidean distance (used in RUNG forward pass)
def pairwise_squared_euclidean(X, Y):
    """
    Adapted from are-gnn-defenses-robust.

    Z_ij = Σ_k (F_ik - F_jk)² = Σ_k F_ik² + F_jk² - 2 F_ik F_jk
    where Σ_k F_ik F_jk = (F F^T)_ij
    """
    squared_X_feat_norms = (X * X).sum(dim=-1)
    squared_Z_feat_norms = (Y * Y).sum(dim=-1)
    pairwise_feat_dot_prods = X @ Y.transpose(-2, -1)
    return (
        -2 * pairwise_feat_dot_prods
        + squared_X_feat_norms[:, None]
        + squared_Z_feat_norms[None, :]
    ).clamp_min(0)

# ---------------------------------------------------------------------------
# New metrics sub-module (Task 4)
# ---------------------------------------------------------------------------
from utils.metrics import (
    compute_estimation_bias,
    compute_bias_curve,
    compute_edge_feature_diff_distribution,
    compute_robust_accuracy,
    compute_clean_and_attacked_accuracy,
)

__all__ = [
    # logging
    "get_log_identifier",
    # graph helpers
    "add_loops",
    "sym_norm",
    "pairwise_squared_euclidean",
    # classification
    "accuracy",
    # python / tensor helpers
    "sub_dict",
    "sp_new_values",
    # estimation-bias metrics
    "compute_estimation_bias",
    "compute_bias_curve",
    "compute_edge_feature_diff_distribution",
    "compute_robust_accuracy",
    "compute_clean_and_attacked_accuracy",
]

