"""
utils package.

Re-exports all symbols from the root utils.py so that existing imports
  ``from utils import add_loops, ...``
continue to work after utils.py is shadowed by this package.

Also exposes the new metrics sub-module:
  ``from utils.metrics import compute_estimation_bias, ...``
"""

import torch

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

