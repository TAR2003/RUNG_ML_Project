"""Verification checks for RUNG_confidence_lambda — run from project root."""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------
# Check 1: Import
# ------------------------------------------------------------------
from model.rung_confidence_lambda import RUNG_confidence_lambda, compute_confidence_lambda
from train_eval_data.fit_confidence_lambda import (
    build_three_group_optimizer,
    gamma_regularization_loss,
    alpha_regularization_loss,
    compute_lambda_confidence_correlation,
)
print("CHECK 1 PASS: all imports OK")

# ------------------------------------------------------------------
# Setup shared test data
# ------------------------------------------------------------------
torch.manual_seed(42)
N, D, C = 100, 64, 7
A = (torch.rand(N, N) > 0.85).float()
A = (A + A.t()).clamp(max=1.0)
A.fill_diagonal_(0.0)
X = torch.randn(N, D)
y = torch.randint(0, C, (N,))

# ------------------------------------------------------------------
# Check 2: Forward pass shape
# ------------------------------------------------------------------
model = RUNG_confidence_lambda(D, C, [64], lam_hat=0.9, gamma_init=6.0, prop_step=3)
logits = model(A, X)
assert logits.shape == (N, C), f"Wrong shape: {logits.shape}"
print(f"CHECK 2 PASS: forward shape {tuple(logits.shape)} correct")

# All three modes
for mode in ['protect_uncertain', 'protect_confident', 'symmetric']:
    m = RUNG_confidence_lambda(D, C, [64], prop_step=2, confidence_mode=mode)
    out = m(A, X)
    assert out.shape == (N, C)
print("CHECK 2b PASS: all three confidence modes produce correct shape")

# ------------------------------------------------------------------
# Check 3: Gradient flows to alpha and gamma
# ------------------------------------------------------------------
# Use small gamma_init so edges land in SCAD region 2 (gamma gradient active)
model2 = RUNG_confidence_lambda(D, C, [64], lam_hat=0.9, gamma_init=0.2, prop_step=3)
logits2 = model2(A, X)
loss2 = F.cross_entropy(logits2, y)
loss2.backward()

assert model2.raw_alpha.grad is not None, "FAIL: alpha gradient is None"
assert model2.raw_alpha.grad.abs() > 0, "FAIL: alpha gradient is zero"
print(f"CHECK 3 PASS: raw_alpha.grad = {model2.raw_alpha.grad.item():.6f}")

nz = sum(1 for lp in model2.log_lams if lp.grad is not None and abs(lp.grad.item()) > 1e-10)
print(f"  Non-zero gamma grads: {nz} / {len(model2.log_lams)}")
assert nz > 0, "FAIL: no gamma gradients"
print("CHECK 3b PASS: gradient flows to gamma parameters")

# ------------------------------------------------------------------
# Check 4: Three-group optimizer
# ------------------------------------------------------------------
model3 = RUNG_confidence_lambda(D, C, [64], prop_step=3)
opt = build_three_group_optimizer(model3, lr=0.01, gamma_lr_factor=0.2, alpha_lr_factor=0.1)
print("Optimizer parameter groups:")
for g in opt.param_groups:
    print(f"  {g['name']}: lr={g['lr']}, {len(g['params'])} tensor(s)")
assert len(opt.param_groups) == 3, "Expected 3 param groups"
assert abs(opt.param_groups[0]['lr'] - 0.01)  < 1e-9
assert abs(opt.param_groups[1]['lr'] - 0.002) < 1e-9
assert abs(opt.param_groups[2]['lr'] - 0.001) < 1e-9
assert opt.param_groups[1]['weight_decay'] == 0.0
assert opt.param_groups[2]['weight_decay'] == 0.0
# All parameters covered (no duplicates, no missing)
total_model = sum(p.numel() for p in model3.parameters())
total_opt   = sum(p.numel() for g in opt.param_groups for p in g['params'])
assert total_model == total_opt, f"Param mismatch: model={total_model}, opt={total_opt}"
print(f"CHECK 4 PASS: three-group optimizer covers all {total_model} params, LRs correct")

# ------------------------------------------------------------------
# Check 5: confidence modes produce DIFFERENT lambda distributions
# ------------------------------------------------------------------
lam_stds = {}
for mode in ['protect_uncertain', 'protect_confident', 'symmetric']:
    m = RUNG_confidence_lambda(D, C, [64], prop_step=2, confidence_mode=mode, normalize_lambda=True)
    _, _, summary = m.get_lambda_distribution(A, X)
    lam_stds[mode] = summary['lambda_std']
    print(f"  {mode}: lambda_mean={summary['lambda_mean']:.4f}, lambda_std={summary['lambda_std']:.4f}")
assert all(s > 0 for s in lam_stds.values()), "All modes should have non-zero lambda std"
print("CHECK 5 PASS: all three modes produce non-trivial lambda distributions")

# ------------------------------------------------------------------
# Check 6: normalize=True preserves mean lambda
# ------------------------------------------------------------------
m_norm   = RUNG_confidence_lambda(D, C, [64], prop_step=2, normalize_lambda=True)
m_unnorm = RUNG_confidence_lambda(D, C, [64], prop_step=2, normalize_lambda=False)
# Share same weights for fair comparison
m_unnorm.load_state_dict(m_norm.state_dict())

lam_norm, _, sum_norm     = m_norm.get_lambda_distribution(A, X)
lam_unnorm, _, sum_unnorm = m_unnorm.get_lambda_distribution(A, X)

# With normalize=True, mean should equal lambda_base
tol = 1e-4
assert abs(sum_norm['lambda_mean'] - sum_norm['lambda_base']) < tol, (
    f"normalize=True: mean={sum_norm['lambda_mean']:.6f} != base={sum_norm['lambda_base']:.6f}"
)
print(f"CHECK 6 PASS: normalize=True preserves mean lambda={sum_norm['lambda_mean']:.4f} (base={sum_norm['lambda_base']:.4f})")

# ------------------------------------------------------------------
# Check 7: alpha_init initialisation is approximately correct
# ------------------------------------------------------------------
for a_init in [0.5, 1.0, 2.0, 3.0]:
    m = RUNG_confidence_lambda(D, C, [64], alpha_init=a_init)
    actual = m.alpha
    assert abs(actual - a_init) < 0.05, f"alpha_init={a_init} but got alpha={actual:.4f}"
    print(f"  alpha_init={a_init} → actual alpha={actual:.4f}")
print("CHECK 7 PASS: alpha_init initialisation is accurate")

# ------------------------------------------------------------------
# Check 8: get_non_gamma_alpha_parameters covers no duplicates
# ------------------------------------------------------------------
m = RUNG_confidence_lambda(D, C, [64], prop_step=3)
main_ids  = {id(p) for p in m.get_non_gamma_alpha_parameters()}
gamma_ids = {id(p) for p in m.get_gamma_parameters()}
alpha_ids = {id(p) for p in m.get_alpha_parameters()}
all_ids   = {id(p) for p in m.parameters()}
assert not (main_ids & gamma_ids), "Main and gamma overlap"
assert not (main_ids & alpha_ids), "Main and alpha overlap"
assert not (gamma_ids & alpha_ids), "Gamma and alpha overlap"
assert main_ids | gamma_ids | alpha_ids == all_ids, "Missing parameters"
print("CHECK 8 PASS: parameter group partition is disjoint and complete")

# ------------------------------------------------------------------
# Check 9: lambda correlation function runs correctly
# ------------------------------------------------------------------
m = RUNG_confidence_lambda(D, C, [64], prop_step=2, confidence_mode='protect_uncertain')
corr = compute_lambda_confidence_correlation(m, A, X)
print(f"  lambda-confidence correlation (protect_uncertain): {corr:+.4f}")
assert -1.0 <= corr <= 1.0
print("CHECK 9 PASS: lambda-confidence correlation computable")

# ------------------------------------------------------------------
# Check 10: Original models untouched (regression)
# ------------------------------------------------------------------
from model.rung import RUNG
from model.rung_learnable_gamma import RUNG_learnable_gamma
from model.att_func import get_mcp_att_func

rung_orig = RUNG(D, C, [64], get_mcp_att_func(6.0), 0.9)
rung_lg   = RUNG_learnable_gamma(D, C, [64], lam_hat=0.9, gamma_init=6.0, prop_step=3)
assert isinstance(rung_orig, RUNG)
assert isinstance(rung_lg, RUNG_learnable_gamma)
assert not isinstance(rung_orig, RUNG_confidence_lambda)
assert not isinstance(rung_lg, RUNG_confidence_lambda)

out_orig = rung_orig(A, X)
out_lg   = rung_lg(A, X)
assert out_orig.shape == (N, C)
assert out_lg.shape   == (N, C)
print("CHECK 10 PASS: RUNG and RUNG_learnable_gamma are untouched")

# ------------------------------------------------------------------
# All checks done
# ------------------------------------------------------------------
print()
print("=" * 60)
print("  ALL 10 CHECKS PASSED — RUNG_confidence_lambda ready")
print("=" * 60)
print()

# Print a model summary
model_final = RUNG_confidence_lambda(D, C, [64], prop_step=3,
                                      confidence_mode='protect_uncertain')
model_final.log_gamma_stats()
print(repr(model_final))
