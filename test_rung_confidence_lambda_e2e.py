"""Quick end-to-end training test for RUNG_confidence_lambda (no seml/gb deps)."""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as nnF
import numpy as np

from model.rung_confidence_lambda import RUNG_confidence_lambda
from train_eval_data.fit_confidence_lambda import (
    fit_confidence_lambda,
    build_three_group_optimizer,
    compute_lambda_confidence_correlation,
)

torch.manual_seed(42)
N, D, C = 150, 32, 5

A = (torch.rand(N, N) > 0.88).float()
A = (A + A.t()).clamp(max=1.0)
A.fill_diagonal_(0.0)
X  = torch.randn(N, D)
y  = torch.randint(0, C, (N,))

idx        = torch.randperm(N)
train_idx  = idx[:80]
val_idx    = idx[80:100]
test_idx   = idx[100:]

print("E2E training test with warmup_epochs=5, max_epoch=20...")

for mode in ['protect_uncertain', 'protect_confident', 'symmetric']:
    model = RUNG_confidence_lambda(
        in_dim=D, out_dim=C, hidden_dims=[32],
        lam_hat=0.9, gamma_init=6.0, prop_step=3,
        confidence_mode=mode, normalize_lambda=True,
    )

    alpha_before = model.alpha
    gammas_before = model.get_learned_gammas()

    fit_confidence_lambda(
        model, A, X, y, train_idx, val_idx,
        lr=0.05, weight_decay=5e-4, max_epoch=20,
        gamma_lr_factor=0.2, alpha_lr_factor=0.1,
        warmup_epochs=5, patience=50,
        log_every=10, grad_clip=1.0,
        alpha_reg_strength=0.001,
    )

    alpha_after  = model.alpha
    gammas_after = model.get_learned_gammas()

    # Alpha should have changed after joint training (warmup freezes it, then it unfreezes)
    alpha_changed  = abs(alpha_after - alpha_before) > 1e-6
    gammas_changed = any(abs(a - b) > 1e-6 for a, b in zip(gammas_after, gammas_before))

    print(f"\n  mode='{mode}':")
    print(f"    alpha  {alpha_before:.4f} → {alpha_after:.4f}  (changed: {alpha_changed})")
    print(f"    gamma0 {gammas_before[0]:.4f} → {gammas_after[0]:.4f}  (changed: {gammas_changed})")

    # Final forward pass
    model.eval()
    with torch.no_grad():
        logits = model(A, X)
        preds  = logits[test_idx].argmax(dim=-1)
        acc    = (preds == y[test_idx]).float().mean().item()
    print(f"    test_acc={acc:.4f}")

    corr = compute_lambda_confidence_correlation(model, A, X)
    print(f"    lambda-conf corr={corr:+.4f}")

    if mode == 'protect_uncertain':
        # Should be negative (uncertain → higher lambda)
        assert corr < 0, f"protect_uncertain should give negative corr, got {corr:.4f}"
    elif mode == 'protect_confident':
        # Should be positive (confident → higher lambda)
        assert corr > 0, f"protect_confident should give positive corr, got {corr:.4f}"

print()
print("=" * 55)
print("  E2E TRAINING TEST PASSED")
print("=" * 55)
