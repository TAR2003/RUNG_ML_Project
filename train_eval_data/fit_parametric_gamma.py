"""
train_eval_data/fit_parametric_gamma.py

Training function for RUNG_parametric_gamma.

Key differences from the standard fit() in train_eval_data/fit.py and from
fit_learnable_gamma.py:

1. Two parameter groups with different learning rates:
   - schedule parameters (log_gamma_0, raw_decay): lower LR (default 0.3× base LR)
     Reason: like gamma in learnable_gamma, these control the SCAD thresholds;
     large steps cause discontinuous jumps in region membership → unstable training.
     Note: We use 0.3× instead of 0.2× because the 2-parameter gradients are stronger
     and more stable than for K separate parameters.
   - all other parameters (MLP weights): standard base LR

2. Two parameters is more stable than K free parameters because:
   - All K layers contribute gradient to both schedule parameters simultaneously
   - Much stronger gradient signal → smoother convergence
   - Forces geometric decay pattern → encodes depth-smoothing hypothesis

3. Gamma value logging every `log_gamma_every` epochs.

4. Gradient clipping for stability (schedule gradients can be large early on).

5. Early stopping on validation accuracy (matching standard robustness evaluation practice).

Everything else is identical to fit_learnable_gamma / standard RUNG training.

Usage:
    from train_eval_data.fit_parametric_gamma import fit_parametric_gamma
    fit_parametric_gamma(model, A, X, y, train_idx, val_idx,
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

def build_optimizer_parametric(
    model,
    lr: float = 5e-2,
    gamma_lr_factor: float = 0.3,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    """
    Build an Adam optimizer with separate learning rates for schedule vs rest.

    Why different LR for schedule parameters?
        The 2 schedule parameters (log_gamma_0, raw_decay) control which SCAD
        region each edge falls in.  A large gradient step can abruptly change
        region membership of many edges simultaneously, producing discontinuous
        loss jumps.  A reduced LR yields smoother, more stable convergence.

    gamma_lr_factor=0.3 means:  schedule_lr = 0.3 × base_lr

    Note: 0.3 is higher than the 0.2 used for learnable_gamma because the
    2-parameter gradients are stronger and more numerically stable.

    Args:
        model:            RUNG_parametric_gamma instance.
        lr:               Base learning rate for non-schedule parameters.
        gamma_lr_factor:  Multiplier for schedule LR. Typical range: [0.2, 0.5].
                          Default 0.3 (higher than learnable_gamma's 0.2 due to
                          stronger gradients from shared parameters).
        weight_decay:     L2 regularisation for non-schedule params.
                          Do NOT apply weight_decay to schedule params (they
                          are thresholds, not feature weights).

    Returns:
        optimizer: torch.optim.Adam with two parameter groups.
    """
    schedule_lr = lr * gamma_lr_factor

    optimizer = torch.optim.Adam(
        [
            {
                "params":       list(model.get_non_gamma_parameters()),
                "lr":           lr,
                "weight_decay": weight_decay,
                "name":         "main_params",
            },
            {
                "params":       model.get_gamma_parameters(),
                "lr":           schedule_lr,
                "weight_decay": 0.0,   # no L2 on schedule parameters
                "name":         "schedule_params",
            },
        ]
    )
    return optimizer


# ============================================================
#  Optional schedule regularisation
# ============================================================

def schedule_regularization_loss(
    model,
    target_gamma_0: float = 3.0,
    target_decay_rate: float = 0.85,
    reg_strength: float = 0.01,
) -> torch.Tensor:
    """
    Soft regularisation that penalises schedule parameters deviating from targets.

    Prevents degenerate solutions:
        gamma_0 → 0:   all edges pruned → model collapses to MLP (no graph)
        gamma_0 → ∞:   no edges pruned → SCAD degenerates
        decay_rate → 0: only layer 0 active → later layers prune everything
        decay_rate → 1: all layers identical → loses layer-specific adaptation

    Loss = reg_strength × [ (log_gamma_0 − log_target_gamma_0)² +
                           (raw_decay − logit_target_decay_rate)² ]

    This is a soft constraint; set reg_strength=0.0 to disable entirely.

    Args:
        model:                RUNG_parametric_gamma instance.
        target_gamma_0:       Centre of regularisation for gamma_0.
        target_decay_rate:    Centre of regularisation for decay_rate.
        reg_strength:         Penalty weight. Typical range: [0.001, 0.1].
                              Keep small relative to cross-entropy loss (~0.01).

    Returns:
        reg_loss: scalar tensor.
    """
    log_target_gamma_0 = float(np.log(max(target_gamma_0, 1e-8)))
    logit_target_decay = float(
        np.log(max(target_decay_rate, 1e-8) / (1.0 - min(target_decay_rate, 1.0 - 1e-8)))
    )

    loss = (
        (model.log_gamma_0 - log_target_gamma_0) ** 2 +
        (model.raw_decay - logit_target_decay) ** 2
    )
    return reg_strength * loss


# ============================================================
#  Main training function
# ============================================================

def fit_parametric_gamma(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    lr: float = 5e-2,
    weight_decay: float = 5e-4,
    max_epoch: int = 300,
    gamma_lr_factor: float = 0.3,
    gamma_reg_strength: float = 0.0,
    patience: int = 100,
    log_gamma_every: int = 50,
    grad_clip: float = 1.0,
    **kwargs,   # absorb any extra kwargs passed by the caller (e.g. from clean.py)
) -> None:
    """
    Training loop for RUNG_parametric_gamma.

    Handles:
    - Two-group optimizer (separate LR for schedule parameters)
    - Early stopping on validation accuracy
    - Optional schedule regularisation
    - Gradient clipping for stability
    - Periodic logging of learned gamma schedule

    Args:
        model:              RUNG_parametric_gamma instance.
        A:                  [N, N] adjacency matrix.
        X:                  [N, D] node feature matrix.
        y:                  [N] integer class labels.
        train_idx:          Training node indices.
        val_idx:            Validation node indices.
        lr:                 Base learning rate (for MLP params).
        weight_decay:       L2 regularisation for MLP params.
        max_epoch:          Maximum training epochs.
        gamma_lr_factor:    LR multiplier for schedule params (0 < f ≤ 1).
                            Default 0.3 (higher than learnable_gamma's 0.2).
        gamma_reg_strength: Schedule regularisation weight (0 = disabled).
        patience:           Early stopping patience in epochs.
        log_gamma_every:    Print gamma schedule every N epochs (0 = never).
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
    optimizer = build_optimizer_parametric(
        model,
        lr=lr,
        gamma_lr_factor=gamma_lr_factor,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None

    print(
        f"\nFit RUNG_parametric_gamma | "
        f"lr={lr}, schedule_lr={lr * gamma_lr_factor:.5f}, "
        f"wd={weight_decay}, epochs={max_epoch}"
    )
    print(
        f"  gamma_reg_strength={gamma_reg_strength}, "
        f"patience={patience}, grad_clip={grad_clip}"
    )
    gammas = model.get_learned_gammas()
    print(
        f"  Initial schedule: gamma_0={model.get_gamma_0_value():.4f}, "
        f"decay_rate={model.get_decay_rate_value():.4f}, "
        f"gammas=[{', '.join(f'{g:.3f}' for g in gammas)}]"
    )

    for epoch in tqdm.trange(max_epoch, desc="fit_parametric_gamma"):

        # ---- train step ----
        model.train()
        optimizer.zero_grad()

        logits   = model(A, X)
        ce_loss  = F.cross_entropy(logits[train_idx], y[train_idx])

        # Optional schedule regularisation
        if gamma_reg_strength > 0.0:
            reg_loss = schedule_regularization_loss(
                model,
                target_gamma_0=model._config["gamma_0_init"],
                target_decay_rate=model._config["decay_rate_init"],
                reg_strength=gamma_reg_strength,
            ).to(device)
            total_loss = ce_loss + reg_loss
        else:
            total_loss = ce_loss

        total_loss.backward()

        # Gradient clipping (helps stabilise schedule updates early in training)
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

        # Periodic schedule logging
        if log_gamma_every > 0 and (epoch + 1) % log_gamma_every == 0:
            gammas = model.get_learned_gammas()
            print(
                f"\n  Epoch {epoch + 1:4d} | "
                f"loss={total_loss.item():.4f} | val={val_acc:.4f} | "
                f"γ₀={model.get_gamma_0_value():.4f}, r={model.get_decay_rate_value():.4f} | "
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
