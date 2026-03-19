"""Test script for 2-parameter gamma schedule differentiability.

Run: python test_parametric.py

Verifies that both log_gamma_0 and raw_decay receive non-zero gradients
when computing the gamma schedule and backpropagating through a fake loss.
"""

import torch
import numpy as np

print("=" * 60)
print("DIFFERENTIABILITY TEST: 2-Parameter Gamma Schedule")
print("=" * 60)

# Simulate the 2-parameter schedule
log_gamma_0 = torch.tensor(1.099, requires_grad=True)   # log(3.0)
raw_decay = torch.tensor(1.735, requires_grad=True)   # logit(0.85)

K = 10
gammas = []
for k in range(K):
    g0 = torch.exp(log_gamma_0)
    r = torch.sigmoid(raw_decay)
    gk = g0 * (r ** k)
    gammas.append(gk)

# Simulate edge differences at each layer (decreasing as features smooth)
# In real code these come from actual node features
fake_loss = sum(
    torch.relu(gk - torch.tensor(0.5 * (K - k) / K))
    for k, gk in enumerate(gammas)
)
fake_loss.backward()

print(f"\nGradient check:")
print(f"  log_gamma_0.grad: {log_gamma_0.grad:.6f}")
print(f"  raw_decay.grad:   {raw_decay.grad:.6f}")

assert log_gamma_0.grad is not None, "ERROR: log_gamma_0.grad is None"
assert raw_decay.grad is not None, "ERROR: raw_decay.grad is None"
assert log_gamma_0.grad.abs() > 0, "ERROR: log_gamma_0.grad is zero"
assert raw_decay.grad.abs() > 0, "ERROR: raw_decay.grad is zero"

print("\n✓ Both parameters receive non-zero gradients!")

# Show the gamma schedule
print("\nGamma schedule:")
log_gamma_0_d = torch.tensor(1.099)
raw_decay_d = torch.tensor(1.735)
for k in range(K):
    g = torch.exp(log_gamma_0_d) * torch.sigmoid(raw_decay_d) ** k
    print(f"  Layer {k:2d}: gamma = {g.item():.4f}")

print("\n" + "=" * 60)
print("PASS: Differentiability test successful!")
print("=" * 60)
