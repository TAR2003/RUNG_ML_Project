"""
train_eval_data/fit_learnable_distance.py

Training function for RUNG_learnable_distance.

Supports THREE distance modes:
    'cosine':     0 new parameters → single-group optimizer (identical to fit_percentile_gamma)
    'projection': learnable MLP → two-group optimizer (optional separate LR for distance)
    'bilinear':   learnable projection → two-group optimizer (optional separate LR for distance)

Key design:
    - For cosine mode: purely single-group training (like RUNG_percentile_gamma)
    - For projection/bilinear modes: offer dist_lr_factor for stable learning
    - Gamma is always percentile-based (no learnable parameters)
    - Automatic parameter grouping based on model.get_distance_parameters()

Training is otherwise identical to fit_percentile_gamma: early stopping,
gradient clipping, validation-best checkpointing, periodic gamma logging.

Usage:
    from train_eval_data.fit_learnable_distance import fit_learnable_distance
    fit_learnable_distance(model, A, X, y, train_idx, val_idx,
                          lr=5e-2, dist_lr_factor=0.5, max_epoch=300)
"""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from utils import accuracy


# ============================================================
#  Optimizer builder — single or two-group depending on distance mode
# ============================================================

def build_optimizer(
    model,
    lr: float = 5e-2,
    dist_lr_factor: float = 0.5,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    """
    Build optimizer with optional separate LR for distance module.

    If model.distance has no parameters (cosine mode):
        → Single-group Adam optimizer (all parameters use same LR)

    If model.distance has parameters (projection/bilinear modes):
        → Two-group optimizer:
            - Group 0: MLP parameters, lr=lr, weight_decay=weight_decay
            - Group 1: distance parameters, lr=lr*dist_lr_factor, weight_decay=0

    Lower LR for distance (dist_lr_factor=0.5) provides stability because
    distance transformations affect the scale of y_ij which affects gamma
    calibration. More conservative learning is appropriate.

    Args:
        model:           RUNG_learnable_distance instance.
        lr:              Base learning rate for all parameters.
        dist_lr_factor:  Multiplier for distance module LR.
                        0.5 = distance LR is half of base LR.
                        Ignored for cosine mode (no distance params).
        weight_decay:    L2 regularisation for non-distance parameters.

    Returns:
        optimizer: torch.optim.Adam, single or two-group.
    """
    dist_params = model.get_distance_parameters()

    if len(dist_params) == 0:
        # Cosine mode: no learnable distance parameters
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    # Projection/bilinear mode: two-group optimizer
    main_params = model.get_non_distance_parameters()

    return torch.optim.Adam([
        {
            'params':       main_params,
            'lr':           lr,
            'weight_decay': weight_decay,
        },
        {
            'params':       dist_params,
            'lr':           lr * dist_lr_factor,
            'weight_decay': 0.0,  # No regularisation on distance params
        },
    ])


# ============================================================
#  Main training function
# ============================================================

def fit_learnable_distance(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    lr: float = 5e-2,
    dist_lr_factor: float = 0.5,
    weight_decay: float = 5e-4,
    max_epoch: int = 300,
    patience: int = 100,
    log_every: int = 50,
    grad_clip: float = 1.0,
    **kwargs,  # absorb any extra kwargs for drop-in replacement
) -> None:
    """
    Training loop for RUNG_learnable_distance.

    Handles:
    - Single or two-group optimizer (depending on distance mode)
    - Early stopping on validation accuracy
    - Gradient clipping for stability
    - Periodic logging of distance metrics and y distribution statistics

    Args:
        model:           RUNG_learnable_distance instance.
        A:               [N, N] adjacency matrix.
        X:               [N, D] node feature matrix.
        y:               [N] integer class labels.
        train_idx:       Training node indices.
        val_idx:         Validation node indices.
        lr:              Learning rate for main (MLP) parameters.
        dist_lr_factor:  LR multiplier for distance module parameters.
                        Only used if model.distance has parameters.
        weight_decay:    L2 regularisation.
        max_epoch:       Maximum training epochs.
        patience:        Early stopping patience in epochs.
        log_every:       Print stats every N epochs (0 = never).
        grad_clip:       Max gradient norm for clipping (0 = disabled).
        **kwargs:        Ignored; allows drop-in replacement of fit().
    """
    # Move data to model's device
    device    = next(model.parameters()).device
    A         = A.to(device)
    X         = X.to(device)
    y         = y.to(device)
    train_idx = train_idx.to(device)
    val_idx   = val_idx.to(device)

    # Build optimizer (single or two-group)
    optimizer = build_optimizer(
        model,
        lr=lr,
        dist_lr_factor=dist_lr_factor,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None

    print(f"\nFit RUNG_learnable_distance")
    print(f"  distance_mode={model.distance_mode}")
    print(f"  lr={lr}, dist_lr_factor={dist_lr_factor}, wd={weight_decay}")
    print(f"  percentile_q={model.percentile_q}")
    if model.use_layerwise_q:
        print(f"  use_layerwise_q=True, percentile_q_late={model.percentile_q_late}")
    print(f"  Parameters: {model.count_parameters()}")
    print(f"    MLP: {sum(p.numel() for p in model.mlp.parameters())}")
    print(f"    Distance: {model.distance.count_parameters()}")
    print(f"  patience={patience}, grad_clip={grad_clip}")

    for epoch in tqdm.trange(max_epoch, desc="fit_learnable_distance"):

        # ---- Train step ----
        model.train()
        optimizer.zero_grad()

        logits = model(A, X)
        loss   = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )

        optimizer.step()

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            val_acc = accuracy(model(A, X)[val_idx], y[val_idx]).item()

        # Early stopping bookkeeping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        # Periodic statistics logging
        if log_every > 0 and (epoch + 1) % log_every == 0:
            gammas = model.get_last_gammas()
            gamma_str = ', '.join(
                f'{g:.2f}' for g in gammas if g is not None
            )
            print(
                f"  Epoch {epoch + 1:4d} | "
                f"loss={loss.item():.4f} | val_acc={val_acc:.4f} | "
                f"gammas=[{gamma_str}]"
            )

            # Log y distribution stats for first and last layers
            if any(s is not None for s in model._last_y_stats):
                y_stats_0 = model._last_y_stats[0]
                y_stats_K = model._last_y_stats[-1]
                if y_stats_0:
                    print(
                        f"    Layer 0: y_mean={y_stats_0[0]:.3f}, "
                        f"y_std={y_stats_0[1]:.3f}, y_max={y_stats_0[2]:.3f}"
                    )
                if y_stats_K and y_stats_K != y_stats_0:
                    print(
                        f"    Layer K: y_mean={y_stats_K[0]:.3f}, "
                        f"y_std={y_stats_K[1]:.3f}, y_max={y_stats_K[2]:.3f}"
                    )

        # Early stopping trigger
        if epoch - best_epoch > patience:
            print(
                f"\n  Early stopping at epoch {epoch + 1} "
                f"(best val={best_val_acc:.4f} at epoch {best_epoch + 1})"
            )
            break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"\nTraining done. Best val acc: {best_val_acc:.4f} "
        f"(epoch {best_epoch + 1})"
    )

    # Show final statistics
    model.eval()
    with torch.no_grad():
        _ = model(A, X)
        model.log_stats()
