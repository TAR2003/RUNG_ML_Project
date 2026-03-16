#!/usr/bin/env python
"""
train_eval_data/fit_percentile_adv_v2.py

FIXED adversarial training for RUNG_percentile_gamma.

This is the corrected version after V1 failed. Changes from V1:
    1. alpha=0.85 (was 0.7) — clean loss dominates for strong base models
    2. train_pgd_steps=100 (was 20) — matches test attack strength
    3. max_epoch=800 (was 300) — enough time at target budget
    4. warmup_epochs=100 (was 0) — MLP stabilizes before adversarial
    5. attack_freq=3 (was 5) — fresher adversarial examples
    6. patience=150 (was 100) — more time to improve at each phase

Key improvements:
    - Explicit adaptive attack check at startup (fails fast if broken)
    - Curriculum V2 with warmup phase
    - Better tracking and reported of training progress
    - Gradient clipping preserved from V1

USAGE:
    from train_eval_data.fit_percentile_adv_v2 import fit_percentile_adv_v2
    
    fit_percentile_adv_v2(
        model, A, X, y, train_idx, val_idx, test_idx,
        attack_fn=pgd_attack,
        alpha=0.85,              # NEW: increased from 0.7
        train_pgd_steps=100,     # NEW: increased from 20
        max_epoch=800,           # NEW: increased from 300
        warmup_epochs=100,       # NEW: was 0
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Tuple
import tqdm


@dataclass
class CurriculumV2:
    """
    Curriculum schedule with warmup phase (NEW in V2).

    Epochs 1-100:    WARMUP — clean training only, no attack
    Epochs 101-150:  Phase 0 — budget = 5%
    Epochs 151-200:  Phase 1 — budget = 10%
    Epochs 201-350:  Phase 2 — budget = 20%
    Epochs 351+:     Phase 3 — budget = target (e.g. 40%)

    The warmup phase prevents early destabilization of the percentile
    gamma mechanism. The long Phase 3 (500+ epochs) ensures the model
    trains sufficiently at the target difficulty.
    """
    warmup_epochs: int = 100
    phase_budgets: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.40])
    phase_epochs: List[Optional[int]] = field(default_factory=lambda: [50, 50, 150, None])
    target_budget: float = 0.40

    def is_warmup(self, epoch: int) -> bool:
        """Returns True if in warmup phase (clean training only)."""
        return epoch <= self.warmup_epochs

    def get_budget(self, epoch: int) -> float:
        """Returns attack budget for given epoch (0 during warmup)."""
        if self.is_warmup(epoch):
            return 0.0

        adj_epoch = epoch - self.warmup_epochs
        cumulative = 0
        for budget, n_epochs in zip(self.phase_budgets, self.phase_epochs):
            if n_epochs is None:
                return budget
            cumulative += n_epochs
            if adj_epoch <= cumulative:
                return budget
        return self.phase_budgets[-1]

    def describe(self) -> str:
        lines = [f"Curriculum V2 (warmup={self.warmup_epochs}):"]
        lines.append(f"  Epochs 1-{self.warmup_epochs}: WARMUP (clean training only)")
        cumulative = self.warmup_epochs
        for b, l in zip(self.phase_budgets, self.phase_epochs):
            if l is None:
                lines.append(f"  Epochs {cumulative+1}+: budget={b:.0%} (rest of training)")
            else:
                lines.append(f"  Epochs {cumulative+1}-{cumulative+l}: budget={b:.0%}")
                cumulative += l
        return "\n".join(lines)


def verify_attack_is_adaptive(
    model: nn.Module,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    attack_fn: Callable,
    budget: float = 0.10,
    n_steps: int = 20,
    device: str = 'cpu',
) -> bool:
    """
    Verify attack is adaptive. Call once at training start.

    If this fails, adversarial training will not work no matter how long
    you run. Stop immediately and fix the attack function.

    Returns:
        True if adaptive, False if not
    """
    print("\n" + "=" * 70)
    print("Verifying attack is truly adaptive...")
    print("=" * 70)

    model = model.to(device)
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)

    n_edges = A.count_nonzero().item() // 2
    budget_edge_num = max(1, int(budget * n_edges))

    # Attack with untrained model
    print("  Attacking untrained model...", end='', flush=True)
    model.eval()
    A_attacked_1 = attack_fn(
        model, A, X, y, train_idx, budget_edge_num, iterations=n_steps
    )
    edges_1 = set(map(tuple, torch.stack(torch.where(A_attacked_1 > 0.5)).T.tolist()))
    print(" done")

    # Train model briefly
    print("  Training model to change weights...", end='', flush=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(50):
        optimizer.zero_grad()
        logits = model(A, X)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
    print(" done")

    # Attack with trained model
    print("  Attacking trained model...", end='', flush=True)
    model.eval()
    A_attacked_2 = attack_fn(
        model, A, X, y, train_idx, budget_edge_num, iterations=n_steps
    )
    edges_2 = set(map(tuple, torch.stack(torch.where(A_attacked_2 > 0.5)).T.tolist()))
    print(" done")

    # Analyze
    original_edges = set(map(tuple, torch.stack(torch.where(A > 0.5)).T.tolist()))
    new_edges_1 = edges_1 - original_edges
    new_edges_2 = edges_2 - original_edges
    overlap = new_edges_1 & new_edges_2

    if len(new_edges_1) == 0:
        print("\n  ERROR: Attack added NO new edges. This is a bug.")
        print("=" * 70 + "\n")
        return False

    overlap_frac = len(overlap) / max(len(new_edges_1), 1)

    print(f"\n  New edges (untrained): {len(new_edges_1)}")
    print(f"  New edges (trained):   {len(new_edges_2)}")
    print(f"  Overlap:               {len(overlap)} ({overlap_frac:.1%})")

    if overlap_frac > 0.85:
        print("\n  FAIL: Attack is NOT adaptive (>85% same edges)")
        print("  This means adversarial training WILL NOT WORK.")
        print("=" * 70 + "\n")
        return False
    else:
        print(f"\n  PASS: Attack is adaptive ({1-overlap_frac:.1%} different edges)")
        print("=" * 70 + "\n")
        return True


def fit_percentile_adv_v2(
    model: nn.Module,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    attack_fn: Callable,
    # Core hyperparameters — FIXED from V1
    alpha: float = 0.85,         # Changed from 0.7
    train_pgd_steps: int = 100,  # Changed from 20
    max_epoch: int = 800,        # Changed from 300
    warmup_epochs: int = 100,    # NEW, was 0
    attack_freq: int = 3,        # Changed from 5
    patience: int = 150,         # Changed from 100
    # Other hyperparameters
    lr: float = 5e-2,
    weight_decay: float = 5e-4,
    log_every: int = 20,
    device: str = 'cpu',
    # Curriculum override (optional)
    curriculum: Optional[CurriculumV2] = None,
    attack_kwargs: Optional[Dict] = None,
) -> Tuple[float, float]:
    """
    FIXED adversarial training with strong baseline.

    This is adversarial training V2 after V1 failed.

    Key changes from V1:
        - alpha=0.85: clean loss dominates (was 0.7, too much adv)
        - train_pgd_steps=100: stronger attacks (was 20, too weak)
        - max_epoch=800: more time (was 300, too short)
        - warmup_epochs=100: stabilization phase (was 0)
        - attack_freq=3: fresher examples (was 5)
        - patience=150: more tolerance per phase (was 100)

    Each change is essential for robust training.

    Args:
        model:              RUNG_percentile_gamma
        A, X, y:            Graph data [dense adjacency, features, labels]
        train_idx, val_idx, test_idx: Node splits
        attack_fn:          PGD attack function
                            Signature: f(model, A, X, y, test_idx, budget_edge_num, iterations)
        alpha:              Clean loss weight (1.0 - alpha = adversarial weight)
        train_pgd_steps:    PGD iterations during training
        max_epoch:          Total training epochs
        warmup_epochs:      Epochs of clean-only training before adversarial
        attack_freq:        Regenerate attack every N epochs
        patience:           Early stopping patience
        lr:                 Learning rate
        weight_decay:       L2 regularization
        log_every:          Logging frequency
        device:             torch device
        curriculum:         Optional CurriculumV2 override
        attack_kwargs:      Extra attack function kwargs

    Returns:
        best_val_acc, test_acc
    """
    if attack_kwargs is None:
        attack_kwargs = {}

    model = model.to(device)
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    # STEP 1: Verify attack is adaptive BEFORE wasting time
    print(f"\nSTEP 1: ADAPTIVE ATTACK VERIFICATION\n")
    is_adaptive = verify_attack_is_adaptive(
        model, A, X, y, train_idx, attack_fn,
        budget=0.10, n_steps=20, device=device
    )
    if not is_adaptive:
        print("FATAL: Attack is not adaptive. Stopping.")
        print("Fix the attack function first, then retry training.")
        return None, None

    # STEP 2: Build curriculum
    if curriculum is None:
        curriculum = CurriculumV2(
            warmup_epochs=warmup_epochs,
            target_budget=0.40,  # Can be overridden per config
        )

    # STEP 3: Build optimizer
    if hasattr(model, 'get_gamma_parameters'):
        optimizer = torch.optim.Adam([
            {
                'params': list(model.get_non_gamma_parameters()),
                'lr': lr,
                'weight_decay': weight_decay,
            },
            {
                'params': list(model.get_gamma_parameters()),
                'lr': lr * 0.3,
                'weight_decay': 0.0,
            },
        ])
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )


    # STEP 4: Print config
    print("\n" + "=" * 70)
    print("RUNG_percentile_adv_v2 — Fixed Adversarial Training".center(70))
    print("=" * 70)
    print(f"  Model:              {model.__class__.__name__}")
    print(f"  alpha:              {alpha} (clean={alpha:.0%}, adv={1-alpha:.0%})")
    print(f"  train_pgd_steps:    {train_pgd_steps}")
    print(f"  attack_freq:        every {attack_freq} epochs")
    print(f"  warmup_epochs:      {warmup_epochs}")
    print(f"  max_epoch:          {max_epoch}")
    print(f"  patience:           {patience}")
    print()
    print(curriculum.describe())
    print("=" * 70 + "\n")

    # STEP 5: Training loop
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    cached_A_pert = A.clone()

    n_edges = A.count_nonzero().item() // 2

    print(f"STEP 2: TRAINING\n")

    for epoch in tqdm.trange(1, max_epoch + 1, desc="train_percentile_adv_v2"):

        budget = curriculum.get_budget(epoch)
        is_warmup = curriculum.is_warmup(epoch)

        # ---- Warmup phase: clean training only ----
        if is_warmup:
            model.train()
            optimizer.zero_grad()
            logits = model(A, X)
            loss = F.cross_entropy(logits[train_idx], y[train_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss = loss.item()
            clean_loss = loss.item()
            adv_loss = 0.0
            phase_name = "WARMUP"

        # ---- Adversarial phase ----
        else:
            # Refresh attack every attack_freq epochs
            if (epoch - warmup_epochs - 1) % attack_freq == 0:
                model.eval()
                budget_edge_num = max(1, int(budget * n_edges))
                t0 = time.time()
                cached_A_pert = attack_fn(
                    model, A, X, y, test_idx, budget_edge_num,
                    iterations=train_pgd_steps,
                    **attack_kwargs
                )
                att_time = time.time() - t0

            # Mixed clean + adversarial training step
            model.train()
            optimizer.zero_grad()

            logits_clean = model(A, X)
            clean_loss = F.cross_entropy(logits_clean[train_idx], y[train_idx])

            logits_adv = model(cached_A_pert, X)
            adv_loss = F.cross_entropy(logits_adv[train_idx], y[train_idx])

            total_loss = alpha * clean_loss + (1.0 - alpha) * adv_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss = total_loss.item()
            clean_loss = clean_loss.item()
            adv_loss = adv_loss.item()
            phase_name = f"budget={budget:.0%}"

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            logits_val = model(A, X)
            val_acc = (logits_val[val_idx].argmax(dim=-1) == y[val_idx]).float().mean().item()

        # ---- Track best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # ---- Logging ----
        if epoch % log_every == 0 or epoch == 1:
            status = (
                f"Ep {epoch:4d} [{phase_name:20s}] | "
                f"L={total_loss:.4f} Lc={clean_loss:.4f} La={adv_loss:.4f} | "
                f"val={val_acc:.4f} (best={best_val_acc:.4f})"
            )
            print(status)

        # ---- Early stopping ----
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best val={best_val_acc:.4f} at epoch {best_epoch}")
            break

    # STEP 6: Final evaluation
    print(f"\n{'='*70}")
    if best_state is None:
        print("No improvement found during training")
        test_acc = 0.0
    else:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits_test = model(A, X)
            test_acc = (logits_test[test_idx].argmax(dim=-1) == y[test_idx]).float().mean().item()

        print(f"Training complete")
        print(f"  Best epoch:   {best_epoch}")
        print(f"  Best val acc: {best_val_acc:.4f}")
        print(f"  Test acc:     {test_acc:.4f}")

    print(f"{'='*70}\n")

    return best_val_acc, test_acc
