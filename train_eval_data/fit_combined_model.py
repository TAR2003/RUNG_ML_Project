"""
train_eval_data/fit_combined_model.py

Training function for RUNG_combined_model.

Uses 2-group optimizer (same pattern as fit_parametric_gamma.py):
    Group 1: MLP weights          → base LR (e.g. 0.05)
    Group 2: schedule + blend     → base LR * gamma_lr_factor (e.g. 0.3)

The 3 new parameters (log_gamma_0, raw_decay, raw_alpha_blend) all go
into Group 2 with the reduced learning rate for stability.
"""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from utils import accuracy


# ============================================================
#  Optimizer builder
# ============================================================

def build_optimizer_combined(
    model,
    lr=0.05,
    gamma_lr_factor=0.3,
    weight_decay=5e-4,
):
    """
    Two-group Adam optimizer for RUNG_combined_model.

    Group 1: MLP encoder parameters  → lr
    Group 2: log_gamma_0, raw_decay, raw_alpha_blend → lr * gamma_lr_factor

    Why lower LR for Group 2:
    - These parameters control SCAD region membership
    - Large steps cause discontinuous transitions → unstable training
    - 0.3x factor matches RUNG_parametric_gamma convention

    Args:
        model:            RUNG_combined_model instance
        lr:               Base learning rate
        gamma_lr_factor:  LR multiplier for schedule/blend params
        weight_decay:     L2 regularization for MLP params only

    Returns:
        optimizer: Adam with two parameter groups
    """
    return torch.optim.Adam([
        {
            'params':       model.get_non_gamma_parameters(),
            'lr':           lr,
            'weight_decay': weight_decay,
            'name':         'mlp',
        },
        {
            'params':       model.get_gamma_parameters(),
            'lr':           lr * gamma_lr_factor,
            'weight_decay': 0.0,
            'name':         'schedule_and_blend',
        },
    ])


def schedule_regularization_loss(
    model,
    target_gamma_0: float = 3.0,
    target_decay_rate: float = 0.85,
    target_alpha_blend: float = 0.5,
    reg_strength: float = 0.01,
) -> torch.Tensor:
    """
    Soft regularisation that penalises schedule parameters deviating from targets.

    Prevents degenerate solutions:
        gamma_0 → 0:   all edges pruned → model collapses to MLP (no graph)
        gamma_0 → ∞:   no edges pruned → SCAD degenerates
        decay_rate → 0: only layer 0 active → later layers prune everything
        decay_rate → 1: all layers identical → loses layer-specific adaptation
        alpha_blend → 0 or 1: loses benefit of blending

    Loss = reg_strength × [ (log_gamma_0 − log_target_gamma_0)² +
                           (raw_decay − logit_target_decay_rate)² +
                           (raw_alpha_blend − logit_target_alpha_blend)² ]

    Args:
        model:                RUNG_combined_model instance
        target_gamma_0:       Centre of regularisation for gamma_0
        target_decay_rate:    Centre of regularisation for decay_rate
        target_alpha_blend:   Centre of regularisation for alpha_blend  
        reg_strength:         Penalty weight

    Returns:
        reg_loss: scalar tensor
    """
    log_target_gamma_0 = float(np.log(max(target_gamma_0, 1e-8)))
    logit_target_decay = float(
        np.log(max(target_decay_rate, 1e-8) / (1.0 - min(target_decay_rate, 1.0 - 1e-8)))
    )
    logit_target_alpha = float(
        np.log(max(target_alpha_blend, 1e-8) / (1.0 - min(target_alpha_blend, 1.0 - 1e-8)))
    )

    loss = (
        (model.log_gamma_0 - log_target_gamma_0) ** 2 +
        (model.raw_decay - logit_target_decay) ** 2 +
        (model.raw_alpha_blend - logit_target_alpha) ** 2
    )
    return reg_strength * loss


# ============================================================
#  Main training function
# ============================================================

def fit_combined_model(
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
    Training loop for RUNG_combined_model.

    Handles:
    - Two-group optimizer (separate LR for schedule parameters)
    - Early stopping on validation accuracy
    - Optional schedule regularisation
    - Gradient clipping for stability
    - Periodic logging of learned gamma schedule + blend weight

    Args:
        model:              RUNG_combined_model instance
        A:                  [N, N] adjacency matrix
        X:                  [N, D] node feature matrix
        y:                  [N] integer class labels
        train_idx:          Training node indices
        val_idx:            Validation node indices
        lr:                 Base learning rate (for MLP params)
        weight_decay:       L2 regularisation for MLP params
        max_epoch:          Maximum training epochs
        gamma_lr_factor:    LR multiplier for schedule params (0 < f ≤ 1)
                            Default 0.3 (higher than learnable_gamma's 0.2)
        gamma_reg_strength: Schedule regularisation weight (0 = disabled)
        patience:           Early stopping patience in epochs
        log_gamma_every:    Print gamma schedule every N epochs (0 = never)
        grad_clip:          Max gradient norm for clipping (0 = disabled)
        **kwargs:           Ignored; allows drop-in replacement of other fit functions
    """
    # Move data to model's device
    device = next(model.parameters()).device
    A         = A.to(device)
    X         = X.to(device)
    y         = y.to(device)
    train_idx = train_idx.to(device)
    val_idx   = val_idx.to(device)

    # Two-group optimizer
    optimizer = build_optimizer_combined(
        model,
        lr=lr,
        gamma_lr_factor=gamma_lr_factor,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None

    print(
        f"\nFit RUNG_combined_model | "
        f"lr={lr}, schedule_lr={lr * gamma_lr_factor:.5f}, "
        f"wd={weight_decay}, epochs={max_epoch}"
    )
    print(
        f"  gamma_reg_strength={gamma_reg_strength}, "
        f"patience={patience}, grad_clip={grad_clip}"
    )
    print(
        f"  Initial schedule: gamma_0={model.gamma_0:.4f}, "
        f"decay_rate={model.decay_rate:.4f}, alpha_blend={model.alpha_blend:.4f}"
    )

    for epoch in tqdm.trange(max_epoch, desc="fit_combined_model"):

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
                target_alpha_blend=model._config["alpha_blend_init"],
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
            print(
                f"\n  Epoch {epoch + 1:4d} | "
                f"loss={total_loss.item():.4f} | val={val_acc:.4f} | "
                f"γ₀={model.gamma_0:.4f}, r={model.decay_rate:.4f}, "
                f"α={model.alpha_blend:.4f}"
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
    model.log_stats()
