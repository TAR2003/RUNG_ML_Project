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
        return f"{model_name}_norm{args.norm}_gamma{args.gamma}"
    
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
        return f"{model_name}_norm{getattr(args, 'norm', 'unknown')}_gamma{getattr(args, 'gamma', 'unknown')}"


# Graph related

def add_loops(A):
    n = A.shape[-1]
    return A + torch.eye(n, device=A.device)

def sym_norm(A):
    Dsq = A.sum(-1).sqrt()
    return A / Dsq / Dsq.unsqueeze(-1)



# metric

def accuracy(
        scores,
        y_true
):
    return (scores.argmax(dim=-1) == y_true).count_nonzero(dim=-1) / y_true.shape[0]

# python helper

def sub_dict(dct, *filter_keys, optional=False):
    # Note: This method raises a KeyError if a desired key is not found, and that is exactly what we want.
    if not optional:
        return {key: dct[key] for key in filter_keys}
    else:
        return {key: dct[key] for key in dct if key in filter_keys}

# tensor helper 

def sp_new_values(t, values):
    out = torch.sparse_coo_tensor(t._indices(), values, t.shape)
    # If the input tensor was coalesced, the output one will be as well since we don't modify the indices.
    if t.is_coalesced():
        with torch.no_grad():
            out._coalesced_(True)
    return out


# model helper

def pairwise_squared_euclidean(X, Y):
    '''
    Adapted from [are_gnn_robust](https://github.com/LoadingByte/are-gnn-defenses-robust)

    $$
    Z_{ij} = \sum_k (F_{ik} - F_{jk})^2 \
        = \sum_k F_{ik}^2 + F_{jk}^2 - 2  F_{ik}  F_{jk}, 
    $$
    where $\sum_k F_{ik}  F_{jk} = (F F^\top)_{ij}$
    The matmul is already implemented efficiently in torch
    '''

    squared_X_feat_norms = (X * X).sum(dim=-1)  # sxfn_i = <X_i|X_i>
    squared_Z_feat_norms = (Y * Y).sum(dim=-1)  # szfn_i = <Z_i|Z_i>
    pairwise_feat_dot_prods = X @ Y.transpose(-2, -1)  # pfdp_ij = <X_i|Z_j> # clever...
    return (-2 * pairwise_feat_dot_prods + squared_X_feat_norms[:, None] + squared_Z_feat_norms[None, :]).clamp_min(0)
