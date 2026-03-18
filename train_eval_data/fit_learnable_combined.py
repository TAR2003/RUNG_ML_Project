"""
train_eval_data/fit_learnable_combined.py

Training function for RUNG_learnable_combined.

Uses two optimizer groups:
- Non-gamma parameters: base lr + weight decay
- Gamma parameters: base lr * gamma_lr_factor (no weight decay)
"""

import torch
import torch.nn.functional as F
import tqdm

from utils import accuracy


def build_optimizer_learnable_combined(
    model,
    lr: float = 5e-2,
    gamma_lr_factor: float = 0.3,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    """Build two-group optimizer with separate LR for gamma parameters."""
    return torch.optim.Adam([
        {
            'params': model.get_non_gamma_parameters(),
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'main_params',
        },
        {
            'params': model.get_gamma_parameters(),
            'lr': lr * gamma_lr_factor,
            'weight_decay': 0.0,
            'name': 'cosine_gamma',
        },
    ])


def fit_learnable_combined(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    lr: float = 5e-2,
    gamma_lr_factor: float = 0.3,
    weight_decay: float = 5e-4,
    max_epoch: int = 300,
    patience: int = 100,
    log_every: int = 50,
    grad_clip: float = 1.0,
    **kwargs,
) -> None:
    """Train RUNG_learnable_combined with early stopping on validation accuracy."""
    device = next(model.parameters()).device
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    optimizer = build_optimizer_learnable_combined(
        model,
        lr=lr,
        gamma_lr_factor=gamma_lr_factor,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    print("\nFit RUNG_learnable_combined")
    print(f"  gamma_mode={model.gamma_mode}")
    print(f"  lr={lr}, gamma_lr_factor={gamma_lr_factor}, wd={weight_decay}")
    print(f"  Parameters: {model.count_parameters()}")
    print(f"  patience={patience}, grad_clip={grad_clip}")

    for epoch in tqdm.trange(max_epoch, desc="fit_learnable_combined"):
        model.train()
        optimizer.zero_grad()

        logits = model(A, X)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = accuracy(model(A, X)[val_idx], y[val_idx]).item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if log_every > 0 and (epoch + 1) % log_every == 0:
            gammas = model.gamma_module.get_all_gammas()
            print(
                f"  Epoch {epoch + 1:4d} | loss={loss.item():.4f} | val={val_acc:.4f} | "
                f"gammas=[{', '.join(f'{g:.3f}' for g in gammas[:3])} ...]"
            )

        if epoch - best_epoch > patience:
            print(
                f"\n  Early stopping at epoch {epoch + 1} "
                f"(best val={best_val_acc:.4f} at epoch {best_epoch + 1})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"\nTraining done. Best val acc: {best_val_acc:.4f} "
        f"(epoch {best_epoch + 1})"
    )

    model.eval()
    with torch.no_grad():
        _ = model(A, X)
        model.log_stats()
