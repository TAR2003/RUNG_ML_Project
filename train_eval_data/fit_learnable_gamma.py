"""
train_eval_data/fit_learnable_gamma.py

Training function for RUNG_learnable_gamma.

Key differences from the standard fit() in train_eval_data/fit.py:

1. Two parameter groups with different learning rates:
   - gamma parameters  (log_lam): lower LR (default 0.2× base LR)
     Reason: gamma controls which edges are pruned; large steps cause
     discontinuous jumps in SCAD region membership → unstable training.
   - all other parameters (MLP weights): standard base LR

2. Optional gamma regularisation: penalises log_lam values that deviate far
   from the initialisation, preventing degenerate solutions
   (gamma → 0 = all edges pruned; gamma → ∞ = no edges pruned).

3. Gamma value logging every `log_gamma_every` epochs.

4. Gradient clipping for stability (gamma gradients can be large early on).

5. Early stopping on validation accuracy (matching standard robustness
   evaluation practice).

Everything else is identical to the standard RUNG/RUNG_new_SCAD training.

Usage:
    from train_eval_data.fit_learnable_gamma import fit_learnable_gamma
    fit_learnable_gamma(model, A, X, y, train_idx, val_idx,
                        lr=5e-2, weight_decay=5e-4, max_epoch=300)
"""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from utils import accuracy


# ============================================================
#  Optimizer builder
# ============================================================

def build_two_group_optimizer(
    model,
    lr: float = 5e-2,
    gamma_lr_factor: float = 0.2,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    """
    Build an Adam optimizer with separate learning rates for gamma vs rest.

    Why different LR for gamma?
        Gamma controls which SCAD region each edge falls in.  A large
        gradient step on gamma can abruptly change the region membership of
        many edges simultaneously, producing discontinuous loss jumps.
        A reduced LR for gamma yields smoother, more stable convergence.

    gamma_lr_factor=0.2 means:  gamma_lr = 0.2 × base_lr

    Args:
        model:            RUNG_learnable_gamma instance.
        lr:               Base learning rate for non-gamma parameters.
        gamma_lr_factor:  Multiplier for gamma LR.  Range: [0.05, 1.0].
        weight_decay:     L2 regularisation for non-gamma params.
                          Do NOT apply weight_decay to gamma (it is a
                          threshold, not a feature weight).

    Returns:
        optimizer: torch.optim.Adam with two parameter groups.
    """
    gamma_lr = lr * gamma_lr_factor

    optimizer = torch.optim.Adam(
        [
            {
                "params":       list(model.get_non_gamma_parameters()),
                "lr":           lr,
                "weight_decay": weight_decay,
                "name":         "main_params",
            },
            {
                "params":       list(model.get_gamma_parameters()),
                "lr":           gamma_lr,
                "weight_decay": 0.0,   # no L2 on threshold parameters
                "name":         "gamma_params",
            },
        ]
    )
    return optimizer


# ============================================================
#  Optional gamma regularisation
# ============================================================

def gamma_regularization_loss(
    model,
    target_lam: float,
    reg_strength: float = 0.01,
) -> torch.Tensor:
    """
    Soft regularisation that penalises log_lam deviating from a target value.

    Prevents degenerate solutions:
        lam → 0 : all edges pruned  → model collapses to MLP (no graph)
        lam → ∞ : no edges pruned   → SCAD degenerates to pure L1

    Loss = reg_strength × Σ_k (log_lam_k − log(target_lam))²

    This is a soft constraint; set reg_strength=0.0 to disable entirely.

    Args:
        model:        RUNG_learnable_gamma instance.
        target_lam:   Centre of the regularisation in lam-space.
                      Default: use gamma_init / a (the initialisation lam).
        reg_strength: Penalty weight.  Typical range: [0.001, 0.1].
                      Keep small relative to cross-entropy loss (~0.01).

    Returns:
        reg_loss: scalar tensor.
    """
    log_target = float(np.log(max(target_lam, 1e-8)))
    reg_loss = sum(
        (log_lam - log_target) ** 2
        for log_lam in model.log_lams
    )
    return reg_strength * reg_loss


# ============================================================
#  Main training function
# ============================================================

def fit_learnable_gamma(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    lr: float = 5e-2,
    weight_decay: float = 5e-4,
    max_epoch: int = 300,
    gamma_lr_factor: float = 0.2,
    gamma_reg_strength: float = 0.0,
    patience: int = 100,
    log_gamma_every: int = 50,
    grad_clip: float = 1.0,
    **kwargs,   # absorb any extra kwargs passed by the caller (e.g. from clean.py)
) -> None:
    """
    Training loop for RUNG_learnable_gamma.

    Handles:
    - Two-group optimizer (separate LR for gamma parameters)
    - Early stopping on validation accuracy
    - Optional gamma regularisation
    - Gradient clipping for stability
    - Periodic logging of learned gamma values

    Args:
        model:              RUNG_learnable_gamma instance.
        A:                  [N, N] adjacency matrix.
        X:                  [N, D] node feature matrix.
        y:                  [N] integer class labels.
        train_idx:          Training node indices.
        val_idx:            Validation node indices.
        lr:                 Base learning rate (for MLP params).
        weight_decay:       L2 regularisation for MLP params.
        max_epoch:          Maximum training epochs.
        gamma_lr_factor:    LR multiplier for gamma params (0 < f ≤ 1).
        gamma_reg_strength: Gamma regularisation weight (0 = disabled).
        patience:           Early stopping patience in epochs.
        log_gamma_every:    Print gamma stats every N epochs (0 = never).
        grad_clip:          Max gradient norm for clipping (0 = disabled).
        **kwargs:           Ignored; allows drop-in replacement of fit().
    """
    # Move data to model's device
    device = next(model.parameters()).device
    A         = A.to(device)
    X         = X.to(device)
    y         = y.to(device)
    train_idx = train_idx.to(device)
    val_idx   = val_idx.to(device)

    # Two-group optimizer
    optimizer = build_two_group_optimizer(
        model,
        lr=lr,
        gamma_lr_factor=gamma_lr_factor,
        weight_decay=weight_decay,
    )

    # For gamma regularisation: target = initialisation lam
    lam_init = model.gamma_init / model.scad_a

    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None

    print(
        f"\nFit RUNG_learnable_gamma | "
        f"lr={lr}, gamma_lr={lr * gamma_lr_factor:.5f}, "
        f"wd={weight_decay}, epochs={max_epoch}"
    )
    print(
        f"  gamma_reg_strength={gamma_reg_strength}, "
        f"patience={patience}, grad_clip={grad_clip}"
    )
    print(f"  Initial gammas: {[f'{g:.3f}' for g in model.get_learned_gammas()]}")

    for epoch in tqdm.trange(max_epoch, desc="fit_learnable_gamma"):

        # ---- train step ----
        model.train()
        optimizer.zero_grad()

        logits   = model(A, X)
        ce_loss  = F.cross_entropy(logits[train_idx], y[train_idx])

        # Optional gamma regularisation
        if gamma_reg_strength > 0.0:
            reg_loss = gamma_regularization_loss(
                model, target_lam=lam_init, reg_strength=gamma_reg_strength
            ).to(device)
            total_loss = ce_loss + reg_loss
        else:
            total_loss = ce_loss

        total_loss.backward()

        # Gradient clipping (helps stabilise gamma updates early in training)
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        # ---- validation ----
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
            gammas = model.get_learned_gammas()
            print(
                f"\n  Epoch {epoch + 1:4d} | "
                f"loss={total_loss.item():.4f} | val={val_acc:.4f} | "
                f"gammas=[{', '.join(f'{g:.2f}' for g in gammas)}]"
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
    model.log_gamma_stats()
