"""
train_eval_data/fit_confidence_lambda.py

Training function for RUNG_confidence_lambda.

Key differences from fit_learnable_gamma.py:

1.  Three parameter groups with potentially different learning rates:
      main:  MLP weights             → base LR
      gamma: log_lam per layer       → lr * gamma_lr_factor  (same as learnable_gamma)
      alpha: raw_alpha sharpness     → lr * alpha_lr_factor  (new, more conservative)

2.  Two-phase training (warmup):
      Phase 1 (warmup_epochs):     freeze gamma and alpha; train MLP only.
                                   Ensures MLP confidences are meaningful
                                   before alpha and gamma calibrate against them.
      Phase 2 (remaining epochs):  unfreeze all; joint optimisation.
    Set warmup_epochs=0 to skip warmup and train everything from the start.

3.  Lambda distribution logging: tracks std(lambda_i) and the
    Pearson correlation between per-node confidence and lambda.
    Use these to confirm the mechanism is active during training.

4.  Alpha regularisation (optional): penalises alpha deviating from 1.0,
    preventing degenerate collapse to alpha≈0 (uniform lambda) or
    alpha→∞ (nearly-binary lambda assignment).

Everything else mirrors fit_learnable_gamma.py exactly.

Usage:
    from train_eval_data.fit_confidence_lambda import fit_confidence_lambda
    fit_confidence_lambda(model, A, X, y, train_idx, val_idx,
                          lr=5e-2, weight_decay=5e-4, max_epoch=300,
                          warmup_epochs=50)
"""

import numpy as np
import torch
import torch.nn.functional as nnF
import tqdm

from utils import accuracy


# ============================================================
#  OPTIMIZER BUILDER (three groups)
# ============================================================

def build_three_group_optimizer(
    model,
    lr: float = 5e-2,
    gamma_lr_factor: float = 0.2,
    alpha_lr_factor: float = 0.1,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    """
    Build an Adam optimizer with three separate learning-rate groups.

    Parameter groups:
        main:  MLP encoder weights  → lr
        gamma: log_lam per layer    → lr * gamma_lr_factor
        alpha: raw_alpha sharpness  → lr * alpha_lr_factor

    Why lower LR for alpha?
        Alpha controls how aggressively confidence is mapped to lambda.
        A large step on alpha can suddenly concentrate lambda on a few nodes
        (if alpha is large, uncertain/confident nodes dominate), causing
        training instability.  Start conservatively; the model will
        fine-tune alpha once MLP confidences are reliable.

    Why lower LR for gamma?
        Identical to RUNG_learnable_gamma: gamma controls SCAD region
        membership for many edges simultaneously; large steps cause
        discontinuous loss jumps.

    Args:
        model:            RUNG_confidence_lambda instance.
        lr:               Base LR for MLP (main) parameters.
        gamma_lr_factor:  LR multiplier for gamma.  Recommended: 0.1–0.3.
        alpha_lr_factor:  LR multiplier for alpha.  Recommended: 0.05–0.2.
        weight_decay:     L2 regularisation for main params.
                          Do NOT apply weight_decay to gamma or alpha.

    Returns:
        optimizer: torch.optim.Adam with three parameter groups.
    """
    optimizer = torch.optim.Adam([
        {
            'params':       list(model.get_non_gamma_alpha_parameters()),
            'lr':           lr,
            'weight_decay': weight_decay,
            'name':         'main_params',
        },
        {
            'params':       list(model.get_gamma_parameters()),
            'lr':           lr * gamma_lr_factor,
            'weight_decay': 0.0,
            'name':         'gamma_params',
        },
        {
            'params':       list(model.get_alpha_parameters()),
            'lr':           lr * alpha_lr_factor,
            'weight_decay': 0.0,
            'name':         'alpha_params',
        },
    ])
    return optimizer


# ============================================================
#  OPTIONAL REGULARISATION LOSSES
# ============================================================

def gamma_regularization_loss(
    model,
    target_lam: float,
    reg_strength: float = 0.01,
) -> torch.Tensor:
    """
    Soft regularisation: penalise log_lam deviating from target value.

    Identical to the version in fit_learnable_gamma.py.
    Prevents degenerate solutions (gamma → 0 or gamma → ∞).

    Loss = reg_strength × Σ_k (log_lam_k − log(target_lam))²

    Args:
        model:        RUNG_confidence_lambda instance.
        target_lam:   Centre value in lam-space (use gamma_init / scad_a).
        reg_strength: Penalty weight (typical range: 0.001–0.1).

    Returns:
        reg_loss: scalar tensor.
    """
    log_target = float(np.log(max(target_lam, 1e-8)))
    reg_loss   = sum(
        (log_lam - log_target) ** 2
        for log_lam in model.log_lams
    )
    return reg_strength * reg_loss


def alpha_regularization_loss(
    model,
    target_alpha: float = 1.0,
    reg_strength: float = 0.001,
) -> torch.Tensor:
    """
    Soft regularisation: penalise alpha deviating from target_alpha.

    Prevents collapse to alpha≈0 (uniform lambda, mechanism inactive)
    or divergence to very large alpha (near-binary lambda, overfits).

    Loss = reg_strength × (alpha − target_alpha)²

    Args:
        model:        RUNG_confidence_lambda instance.
        target_alpha: Centre value (default 1.0 = linear confidence map).
        reg_strength: Penalty weight (default 0.001 is conservative).
                      Set to 0.0 to disable.

    Returns:
        reg_loss: scalar tensor.
    """
    current_alpha = model.get_alpha_tensor()
    return reg_strength * (current_alpha - target_alpha) ** 2


# ============================================================
#  DIAGNOSTIC UTILITIES
# ============================================================

def compute_lambda_confidence_correlation(model, A, X) -> float:
    """
    Compute Pearson correlation between per-node confidence and lambda.

    Interpretation by mode:
        'protect_uncertain'  → expect NEGATIVE correlation
                               (uncertain = high lambda)
        'protect_confident'  → expect POSITIVE correlation
                               (confident = high lambda)
        'symmetric'          → expect correlation near 0 (U-shaped relation)

    Args:
        model:  RUNG_confidence_lambda (eval mode recommended).
        A:      [N, N] adjacency matrix.
        X:      [N, D] node features.

    Returns:
        correlation: float in [-1, 1].
    """
    lambda_vals, conf_vals, _ = model.get_lambda_distribution(A, X)

    lam_c = lambda_vals - lambda_vals.mean()
    con_c = conf_vals   - conf_vals.mean()
    denom = (lam_c.std() * con_c.std()).clamp(min=1e-8)
    corr  = (lam_c * con_c).mean() / denom

    return corr.item()


# ============================================================
#  MAIN TRAINING FUNCTION
# ============================================================

def fit_confidence_lambda(
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
    alpha_lr_factor: float = 0.1,
    gamma_reg_strength: float = 0.0,
    alpha_reg_strength: float = 0.001,
    warmup_epochs: int = 50,
    patience: int = 100,
    log_every: int = 50,
    grad_clip: float = 1.0,
    **kwargs,   # absorb extra kwargs for drop-in compatibility
) -> None:
    """
    Training loop for RUNG_confidence_lambda.

    Two-phase training:
    ─────────────────────────────────────────────────────────────
    Phase 1 — MLP warmup  (epochs 1 … warmup_epochs):
        Gamma and alpha are FROZEN.  Only MLP weights update.
        Purpose: MLP learns to produce meaningful features and
        pre-aggregation confidences before gamma/alpha calibrate.
        Without warmup, alpha may converge to ≈0 early (making all
        lambdas equal) while MLP confidences are still random noise.

    Phase 2 — Joint training  (epochs warmup_epochs+1 … max_epoch):
        All parameters training with their respective learning rates.
        Gamma and alpha calibrate against now-meaningful confidences.

    Set warmup_epochs=0 to skip Phase 1 and train jointly from epoch 1.
    ─────────────────────────────────────────────────────────────

    Training loss:
        L = CE(logits, y)
            + gamma_reg_strength × Σ_k (log_lam_k − log(lam_init))²
            + alpha_reg_strength × (alpha − 1.0)²

    Monitoring:
        Every log_every epochs, prints:
          - CE + reg loss
          - validation accuracy
          - current alpha value
          - std(lambda_i) — non-zero means redistribution is happening
          - Pearson corr(lambda, conf) — reveals mode-specific patterns

    Args:
        model:              RUNG_confidence_lambda instance.
        A:                  [N, N] clean adjacency matrix.
        X:                  [N, D] node feature matrix.
        y:                  [N] integer class labels.
        train_idx:          Training node indices.
        val_idx:            Validation node indices.
        lr:                 Base learning rate for MLP weights.
        weight_decay:       L2 regularisation for MLP weights.
        max_epoch:          Maximum training epochs.
        gamma_lr_factor:    LR multiplier for gamma parameters (recommended 0.2).
        alpha_lr_factor:    LR multiplier for alpha parameter (recommended 0.1).
        gamma_reg_strength: Gamma regularisation weight (0 = disabled).
        alpha_reg_strength: Alpha regularisation weight (default 0.001).
        warmup_epochs:      Epochs to train only MLP before joint training.
                            Set to 0 to skip warmup.
        patience:           Early stopping patience in epochs.
        log_every:          Print diagnostics every N epochs (0 = never).
        grad_clip:          Max gradient norm for clipping (0 = disabled).
        **kwargs:           Ignored; allows drop-in replacement of fit().
    """
    # Move all tensors to the model's device
    device    = next(model.parameters()).device
    A         = A.to(device)
    X         = X.to(device)
    y         = y.to(device)
    train_idx = train_idx.to(device)
    val_idx   = val_idx.to(device)

    # Three-group optimizer
    optimizer = build_three_group_optimizer(
        model,
        lr               = lr,
        gamma_lr_factor  = gamma_lr_factor,
        alpha_lr_factor  = alpha_lr_factor,
        weight_decay     = weight_decay,
    )

    # Gamma regularisation target: use initialisation lam so gamma stays near init
    lam_init = model.gamma_init / model.scad_a

    # ---- Phase 1: freeze gamma + alpha ----
    if warmup_epochs > 0:
        for p in model.get_gamma_parameters():
            p.requires_grad_(False)
        for p in model.get_alpha_parameters():
            p.requires_grad_(False)

    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None

    print(
        f"\nFit RUNG_confidence_lambda | "
        f"lr={lr}, gamma_lr={lr * gamma_lr_factor:.5f}, "
        f"alpha_lr={lr * alpha_lr_factor:.5f}, "
        f"wd={weight_decay}, epochs={max_epoch}"
    )
    print(
        f"  mode={model.confidence_mode}, normalize={model.normalize_lambda}, "
        f"warmup_epochs={warmup_epochs}"
    )
    print(
        f"  gamma_reg={gamma_reg_strength}, alpha_reg={alpha_reg_strength}, "
        f"patience={patience}, grad_clip={grad_clip}"
    )
    print(f"  Initial gammas: {[f'{g:.3f}' for g in model.get_learned_gammas()]}")
    print(f"  Initial alpha:  {model.alpha:.4f}")
    if warmup_epochs > 0:
        print(f"\nPhase 1: MLP warmup for {warmup_epochs} epochs "
              f"(gamma + alpha frozen)")

    for epoch in tqdm.trange(max_epoch, desc="fit_confidence_lambda"):

        # ---- Phase 2: unfreeze gamma + alpha after warmup ----
        if epoch == warmup_epochs and warmup_epochs > 0:
            for p in model.get_gamma_parameters():
                p.requires_grad_(True)
            for p in model.get_alpha_parameters():
                p.requires_grad_(True)
            print(f"\nPhase 2: Joint training (all params unfrozen)\n")

        # ---- Training step ----
        model.train()
        optimizer.zero_grad()

        logits  = model(A, X)
        ce_loss = nnF.cross_entropy(logits[train_idx], y[train_idx])

        # Optional gamma regularisation (identical to fit_learnable_gamma)
        total_loss = ce_loss
        if gamma_reg_strength > 0.0:
            g_reg       = gamma_regularization_loss(
                model, target_lam=lam_init, reg_strength=gamma_reg_strength
            ).to(device)
            total_loss  = total_loss + g_reg

        # Optional alpha regularisation
        if alpha_reg_strength > 0.0:
            a_reg       = alpha_regularization_loss(
                model, target_alpha=1.0, reg_strength=alpha_reg_strength
            ).to(device)
            total_loss  = total_loss + a_reg

        total_loss.backward()

        # Gradient clipping — important: alpha gradients can spike
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

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

        # ---- Periodic diagnostic logging ----
        if log_every > 0 and (epoch + 1) % log_every == 0:
            _, _, lam_summary = model.get_lambda_distribution(A, X)
            corr = compute_lambda_confidence_correlation(model, A, X)

            print(
                f"\n  Epoch {epoch + 1:4d} | "
                f"loss={total_loss.item():.4f} | val={val_acc:.4f} | "
                f"α={model.alpha:.3f} | "
                f"λ_std={lam_summary['lambda_std']:.4f} | "
                f"λ-conf_corr={corr:+.3f} | "
                f"gammas=[{', '.join(f'{g:.2f}' for g in model.get_learned_gammas())}]"
            )

        # Early stopping
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
