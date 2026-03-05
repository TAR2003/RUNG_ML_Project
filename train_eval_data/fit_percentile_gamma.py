"""
train_eval_data/fit_percentile_gamma.py

Training function for RUNG_percentile_gamma.

Key differences from fit_learnable_gamma.py:

1.  Single optimizer group — gamma has no parameters, so there is no
    need for a two-group optimizer.  This is strictly simpler than
    RUNG_learnable_gamma training.

2.  Gamma logging every log_gamma_every epochs: runs an eval forward pass
    to capture how data-driven gammas evolve as MLP weights change.
    Tracks the same information as fit_learnable_gamma.py for direct
    comparison, but the values here come from quantile computation,
    not gradient updates.

3.  Everything else is identical to fit_learnable_gamma.py:
    early stopping, gradient clipping, validation-best checkpointing.

The training is simpler because there are fewer things to tune:
    RUNG_learnable_gamma:  tune LR, gamma_lr_factor, gamma_reg_strength
    RUNG_percentile_gamma: tune LR only (gamma is fully automatic)

Usage:
    from train_eval_data.fit_percentile_gamma import fit_percentile_gamma
    fit_percentile_gamma(model, A, X, y, train_idx, val_idx,
                         lr=5e-2, weight_decay=5e-4, max_epoch=300)
"""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from utils import accuracy


# ============================================================
#  Optimizer builder — single group (simpler than learnable_gamma)
# ============================================================

def build_optimizer(
    model,
    lr: float = 5e-2,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    """
    Build a single-group Adam optimizer for RUNG_percentile_gamma.

    Because gamma has no parameters in this model, all trainable
    parameters belong to the MLP.  A single LR group suffices.

    Contrast with RUNG_learnable_gamma which needs a two-group optimizer
    (separate, lower LR for gamma) to avoid unstable gradient steps on
    the SCAD threshold.  That complexity is entirely absent here.

    Args:
        model:        RUNG_percentile_gamma instance.
        lr:           Learning rate for all parameters (MLP only).
        weight_decay: L2 regularisation.

    Returns:
        optimizer: torch.optim.Adam with a single parameter group.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


# ============================================================
#  Main training function
# ============================================================

def fit_percentile_gamma(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    lr: float = 5e-2,
    weight_decay: float = 5e-4,
    max_epoch: int = 300,
    patience: int = 100,
    log_gamma_every: int = 50,
    grad_clip: float = 1.0,
    **kwargs,   # absorb any extra kwargs for drop-in replacement of fit()
) -> None:
    """
    Training loop for RUNG_percentile_gamma.

    Handles:
    - Single-group Adam optimizer (simpler than learnable_gamma)
    - Early stopping on validation accuracy
    - Gradient clipping for stability
    - Periodic logging of percentile gamma values across layers

    Args:
        model:           RUNG_percentile_gamma instance.
        A:               [N, N] adjacency matrix.
        X:               [N, D] node feature matrix.
        y:               [N] integer class labels.
        train_idx:       Training node indices.
        val_idx:         Validation node indices.
        lr:              Learning rate (for MLP params only).
        weight_decay:    L2 regularisation.
        max_epoch:       Maximum training epochs.
        patience:        Early stopping patience in epochs.
        log_gamma_every: Print gamma stats every N epochs (0 = never).
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

    # Single-group optimizer — no gamma LR tuning needed
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None

    print(
        f"\nFit RUNG_percentile_gamma | "
        f"lr={lr}, wd={weight_decay}, epochs={max_epoch}"
    )
    print(
        f"  percentile_q={model.percentile_q}, "
        f"use_layerwise_q={model.use_layerwise_q}"
    )
    if model.use_layerwise_q:
        print(f"  percentile_q_late={model.percentile_q_late}")
    print(f"  Parameters: {model.count_parameters()} (no gamma params)")
    print(f"  patience={patience}, grad_clip={grad_clip}")

    for epoch in tqdm.trange(max_epoch, desc="fit_percentile_gamma"):

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

        # Periodic gamma logging
        if log_gamma_every > 0 and (epoch + 1) % log_gamma_every == 0:
            gammas = model.get_last_gammas()
            if any(g is not None for g in gammas):
                print(
                    f"\n  Epoch {epoch + 1:4d} | "
                    f"loss={loss.item():.4f} | val={val_acc:.4f} | "
                    f"gammas=[{', '.join(f'{g:.2f}' for g in gammas if g is not None)}]"
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

    # Show final gamma profile from a clean eval forward pass
    model.eval()
    with torch.no_grad():
        _ = model(A, X)
    model.log_gamma_stats()
