"""
train_eval_data/fit_combined.py

Training function for RUNG_combined.

RUNG_combined combines:
    - Cosine distance (scale-invariant, no new parameters)
    - Percentile gamma (auto-adaptive, no new parameters)

Training is identical to RUNG_percentile_gamma because:
    1. No learnable distance parameters (cosine is free)
    2. No gamma parameters (percentile is computed from data)
    3. Only MLP parameters are optimized via standard Adam

Simple single-group optimizer (like RUNG_percentile_gamma).
Early stopping, gradient clipping, validation-best checkpointing.

The only tuning required is learning rate (same as all RUNG models).
Percentile_q can be tuned but is typically set to 0.75 (same as parents).
"""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from utils import accuracy


# ============================================================
#  Optimizer builder — single group (cosine + percentile have no params)
# ============================================================

def build_optimizer(
    model,
    lr: float = 5e-2,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    """
    Build a single-group Adam optimizer for RUNG_combined.

    Because both cosine distance and percentile gamma have zero parameters,
    only MLP parameters are trainable. A single LR group suffices.

    Args:
        model:        RUNG_combined instance.
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

def fit_combined(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor = None,
    max_epoch: int = 1000,
    lr: float = 0.05,
    weight_decay: float = 5e-4,
    early_stopping_patience: int = 100,
    log_every: int = 10,
    log_gamma_every: int = 50,
    device: str = 'cpu',
    verbose: bool = True,
) -> dict:
    """
    Train RUNG_combined for node classification.

    Training loop for RUNG_combined with early stopping based on
    validation accuracy.

    Args:
        model:                   RUNG_combined instance
        A:                       [N, N] adjacency matrix
        X:                       [N, D] node feature matrix
        y:                       [N] node labels (integer class indices)
        train_idx:               [T] training node indices
        val_idx:                 [V] validation node indices
        test_idx:                [Te] test node indices (optional)
        max_epoch:               Maximum training epochs (default 1000)
        lr:                      Learning rate (default 0.05)
        weight_decay:            L2 regularization (default 5e-4)
        early_stopping_patience: Patience for early stopping (default 100)
        log_every:               Log frequency (default 10)
        log_gamma_every:         Gamma logging frequency (default 50)
        device:                  torch device (default 'cpu')
        verbose:                 Print progress (default True)

    Returns:
        dict with keys:
            'best_val_acc':      Best validation accuracy achieved
            'best_epoch':        Epoch at which best validation was achieved
            'train_accs':        List of train accuracies per epoch
            'val_accs':          List of validation accuracies per epoch
            'test_acc':          Final test accuracy (if test_idx provided)
            'gammas':            Final gamma values per layer
    """
    model = model.to(device)
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    train_accs = []
    val_accs = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"{'Fitting RUNG_combined':^70}")
        print(f"{'='*70}")
        print(f"  Learning rate:     {lr}")
        print(f"  Weight decay:      {weight_decay}")
        print(f"  Percentile q:      {model.percentile_q}")
        print(f"  Distance:          cosine")
        print(f"  Num layers:        {model.prop_layer_num}")
        print(f"  Parameters:        {model.count_parameters()}")
        print(f"{'='*70}\n")

    for epoch in range(1, max_epoch + 1):
        # ---- Training step ----
        model.train()
        optimizer.zero_grad()

        logits = model(A, X)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---- Validation step ----
        model.eval()
        with torch.no_grad():
            logits_val = model(A, X)
            train_acc = accuracy(logits_val[train_idx], y[train_idx])
            val_acc = accuracy(logits_val[val_idx], y[val_idx])

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # ---- Early stopping logic ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # ---- Logging ----
        if verbose and epoch % log_every == 0:
            print(f"Epoch {epoch:4d} | "
                  f"loss={loss.item():7.4f} | "
                  f"train={train_acc:.4f} | "
                  f"val={val_acc:.4f} | "
                  f"best_val={best_val_acc:.4f}")

        # ---- Gamma logging (less frequent) ----
        if verbose and epoch % log_gamma_every == 0:
            with torch.no_grad():
                _ = model(A, X)
            gammas_str = ", ".join(f"{g:.4f}" for g in model._last_gammas[:5])
            print(f"         Gammas (first 5 layers): {gammas_str}")

        # ---- Early stopping ----
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(patience={early_stopping_patience})")
            break

    # ---- Load best checkpoint ----
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final evaluation ----
    model.eval()
    with torch.no_grad():
        logits_final = model(A, X)
        final_val_acc = accuracy(logits_final[val_idx], y[val_idx])
        final_train_acc = accuracy(logits_final[train_idx], y[train_idx])
        final_test_acc = None
        if test_idx is not None:
            final_test_acc = accuracy(logits_final[test_idx], y[test_idx])

    if verbose:
        print(f"\n{'='*70}")
        print(f"Training complete!")
        print(f"  Best epoch:        {best_epoch}")
        print(f"  Best val acc:      {best_val_acc:.4f}")
        print(f"  Final train acc:   {final_train_acc:.4f}")
        print(f"  Final val acc:     {final_val_acc:.4f}")
        if final_test_acc is not None:
            print(f"  Final test acc:    {final_test_acc:.4f}")
        print(f"{'='*70}\n")

    # ---- Log final stats ----
    if verbose:
        model.log_stats()

    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_acc': final_test_acc,
        'gammas': model.get_last_gammas(),
    }
