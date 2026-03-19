"""
train_eval_data/fit_homophily_adaptive.py

Training function for RUNG_homophily_adaptive.

This follows the same training pattern used by percentile-based variants:
single optimizer group, standard CE loss, early stopping, and gradient clipping.
"""

import torch
import torch.nn.functional as F
import tqdm

from utils import accuracy


def build_optimizer(
    model,
    lr: float = 5e-2,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def fit_homophily_adaptive(
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
    log_every: int = 50,
    grad_clip: float = 1.0,
    **kwargs,
) -> None:
    device = next(model.parameters()).device
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    print(
        f"\nFit RUNG_homophily_adaptive | "
        f"lr={lr}, wd={weight_decay}, epochs={max_epoch}"
    )
    print(
        f"  percentile_q={model.percentile_q}, q_relax={model.q_relax}, "
        f"q_max={model.q_max}, homophily_mode={model.homophily_mode}"
    )
    print(f"  Parameters: {model.count_parameters()}")
    print(f"  patience={patience}, grad_clip={grad_clip}")

    for epoch in tqdm.trange(max_epoch, desc="fit_homophily_adaptive"):
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
            print(
                f"  Epoch {epoch + 1:4d} | "
                f"loss={loss.item():.4f} | val_acc={val_acc:.4f} | "
                f"h_mean={model._last_h_mean if model._last_h_mean is not None else float('nan'):.4f} | "
                f"q_mean={model._last_q_mean if model._last_q_mean is not None else float('nan'):.4f}"
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
