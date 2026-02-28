"""
Evaluation metrics for RUNG robustness analysis.

Implements the estimation bias metrics from Section 4.3 and Figure 6 of the
RUNG paper (NeurIPS 2024), plus helpers for feature difference distributions.

All functions work with the codebase's native format:
    A : [N, N] dense float32 adjacency matrix
    X : [N, D] float32 node feature matrix
    y : [N] int64 label tensor
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Estimation bias (Section 4.3 / Figure 6)
# ---------------------------------------------------------------------------

def compute_estimation_bias(
    model: torch.nn.Module,
    A_clean: torch.Tensor,
    X: torch.Tensor,
    A_attacked: torch.Tensor,
    device: torch.device,
    return_per_node: bool = False,
) -> Tuple:
    """
    Compute estimation bias of graph signal aggregation under attack.

    Bias = Σ_{i ∈ V} ||f_i - f_i*||_2²

    where:
        f_i*  = aggregated feature of node i on the CLEAN graph
        f_i   = aggregated feature of node i on the ATTACKED graph

    A low bias means the model's aggregated features are not distorted by
    adversarial edges — the key property RUNG is designed to achieve
    (Section 4.3 of the paper).

    Args:
        model:          Trained RUNG (or any GNN) model with
                        `get_aggregated_features(A, X)` method
        A_clean:        [N, N] clean adjacency matrix (float32)
        X:              [N, D] node feature matrix (float32)
        A_attacked:     [N, N] adversarially perturbed adjacency matrix
        device:         torch.device for computation
        return_per_node: If True, also return per-node bias vector

    Returns:
        bias_total:     float, Σ_i ||f_i - f_i*||²
        bias_mean:      float, mean bias per node
        bias_per_node:  [Optional] Tensor [N] (returned when return_per_node=True)
    """
    model.eval()

    A_clean    = A_clean.to(device)
    A_attacked = A_attacked.to(device)
    X          = X.to(device)

    with torch.no_grad():
        f_clean    = model.get_aggregated_features(A_clean, X)     # [N, C]
        f_attacked = model.get_aggregated_features(A_attacked, X)  # [N, C]

    diff          = f_clean - f_attacked                           # [N, C]
    bias_per_node = (diff ** 2).sum(dim=-1)                        # [N]

    bias_total = bias_per_node.sum().item()
    bias_mean  = bias_per_node.mean().item()

    if return_per_node:
        return bias_total, bias_mean, bias_per_node.cpu()
    return bias_total, bias_mean


def compute_bias_curve(
    model: torch.nn.Module,
    A_clean: torch.Tensor,
    X: torch.Tensor,
    attacked_graphs_by_budget: Dict[float, torch.Tensor],
    device: torch.device,
) -> Dict[float, Dict[str, float]]:
    """
    Compute estimation bias curve across attack budgets.

    Replicates Figure 6 from the RUNG paper, extended to all penalty types.

    Args:
        model:                      Trained RUNG model
        A_clean:                    [N, N] clean adjacency matrix
        X:                          [N, D] node feature matrix
        attacked_graphs_by_budget:  Dict {budget_fraction → A_attacked [N, N]}
                                    e.g. {0.05: A_5pct, 0.10: A_10pct, ...}
        device:                     torch.device

    Returns:
        bias_curve: Dict {budget_fraction → {'bias_total': float, 'bias_mean': float}}

    Example:
        budgets = {0.05: A_5pct, 0.10: A_10pct, 0.20: A_20pct}
        curve = compute_bias_curve(model, A_clean, X, budgets, device)
        # curve[0.10]['bias_total'] → float
    """
    bias_curve = {}

    for budget, A_attacked in sorted(attacked_graphs_by_budget.items()):
        bias_total, bias_mean = compute_estimation_bias(
            model, A_clean, X, A_attacked, device
        )
        bias_curve[budget] = {
            'bias_total': bias_total,
            'bias_mean':  bias_mean,
        }
        print(f"  Budget {budget*100:.1f}%: "
              f"total_bias={bias_total:.4f}, mean_bias={bias_mean:.6f}")

    return bias_curve


# ---------------------------------------------------------------------------
# Feature difference distribution (Section 4.3 / Figure 7)
# ---------------------------------------------------------------------------

def compute_edge_feature_diff_distribution(
    model: torch.nn.Module,
    A: torch.Tensor,
    X: torch.Tensor,
    device: torch.device,
    num_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute distribution of normalized feature differences on edges.

    Replicates Figure 7 from the RUNG paper, which shows that RUNG forces
    adversarial edges to have small feature differences (the attacker must
    add edges between similar nodes, reducing attack effectiveness).

    The distribution is of:
        y_ij = ||f_i/√d_i - f_j/√d_j||_2  for each edge (i, j)

    Args:
        model:    Trained RUNG model with `get_aggregated_features(A, X)`
        A:        [N, N] adjacency matrix (typically attacked graph)
        X:        [N, D] node feature matrix
        device:   torch.device
        num_bins: Number of histogram bins

    Returns:
        hist_values:  np.ndarray [num_bins] — normalized histogram (density)
        bin_edges:    np.ndarray [num_bins+1] — bin boundary values
        mean_diff:    float, mean feature difference over all edges
    """
    model.eval()

    A_dev = A.to(device)
    X_dev = X.to(device)

    with torch.no_grad():
        features = model.get_aggregated_features(A_dev, X_dev)  # [N, C]

    # Add self-loops to get degree consistent with RUNG's own preprocessing
    A_loops = A_dev + torch.eye(A_dev.shape[0], device=device)
    degrees = A_loops.sum(-1)                                    # [N]
    degrees_safe = degrees.clamp(min=1.0)

    # Normalised features: f_i / √d_i
    f_norm = features / degrees_safe.sqrt().unsqueeze(-1)        # [N, C]

    # Gather edge endpoints
    src, dst = A_dev.nonzero(as_tuple=True)

    diffs = (f_norm[src] - f_norm[dst]).norm(dim=-1)             # [num_edges]
    diffs_np = diffs.cpu().numpy()

    hist_values, bin_edges = np.histogram(diffs_np, bins=num_bins, density=True)
    mean_diff = float(diffs_np.mean())

    return hist_values, bin_edges, mean_diff


# ---------------------------------------------------------------------------
# Accuracy utilities (complements utils.py)
# ---------------------------------------------------------------------------

def compute_robust_accuracy(
    model: torch.nn.Module,
    A_attacked: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    test_idx: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Evaluate model accuracy on an adversarially attacked graph.

    Args:
        model:      Trained GNN model
        A_attacked: [N, N] perturbed adjacency matrix
        X:          [N, D] feature matrix
        y:          [N] ground-truth labels
        test_idx:   [T] test node indices
        device:     torch.device

    Returns:
        accuracy: float in [0, 1]
    """
    model.eval()

    with torch.no_grad():
        logits = model(A_attacked.to(device), X.to(device))     # [N, C]
        preds  = logits.argmax(dim=-1)                           # [N]
        correct = (preds[test_idx] == y[test_idx].to(device)).float()

    return correct.mean().item()


def compute_clean_and_attacked_accuracy(
    model: torch.nn.Module,
    A_clean: torch.Tensor,
    A_attacked: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    test_idx: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Return (clean_accuracy, attacked_accuracy) for a single model.

    Convenience wrapper for `compute_robust_accuracy`.
    """
    clean_acc   = compute_robust_accuracy(model, A_clean,    X, y, test_idx, device)
    attacked_acc = compute_robust_accuracy(model, A_attacked, X, y, test_idx, device)
    return clean_acc, attacked_acc
