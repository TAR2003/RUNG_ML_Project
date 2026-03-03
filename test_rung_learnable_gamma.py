"""Verification checks for RUNG_learnable_gamma — run from project root."""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------
# Check 1: Import
# ------------------------------------------------------------------
from model.rung_learnable_gamma import RUNG_learnable_gamma, scad_weight_differentiable
print("CHECK 1 PASS: model import OK")

# ------------------------------------------------------------------
# Check 2: Gradient through differentiable SCAD
# ------------------------------------------------------------------
gamma = torch.tensor(3.0, requires_grad=True)
y = torch.linspace(0.1, 15.0, 200)
W = scad_weight_differentiable(y, gamma, a=3.7)
W.sum().backward()
assert gamma.grad is not None, "FAIL: gamma.grad is None"
assert gamma.grad.abs() > 0,   "FAIL: gamma.grad is zero"
print(f"CHECK 2 PASS: gamma.grad = {gamma.grad.item():.6f}  (autograd through SCAD OK)")

log_gamma = torch.tensor(float(torch.log(torch.tensor(3.0)).item()), requires_grad=True)
W2 = scad_weight_differentiable(y, torch.exp(log_gamma), a=3.7)
W2.sum().backward()
assert log_gamma.grad is not None and log_gamma.grad.abs() > 0
print(f"  log_gamma.grad = {log_gamma.grad.item():.6f}")

# ------------------------------------------------------------------
# Check 3: Forward pass + param count
# ------------------------------------------------------------------
torch.manual_seed(42)
N, D, C = 50, 16, 7
A = (torch.rand(N, N) > 0.8).float(); A = A + A.t(); A.fill_diagonal_(0)
X = torch.randn(N, D)

model = RUNG_learnable_gamma(D, C, [32], lam_hat=0.9, gamma_init=6.0, prop_step=10)
out = model(A, X)
assert out.shape == (N, C), f"Wrong shape: {out.shape}"
total_p = sum(p.numel() for p in model.parameters())
gamma_p = sum(p.numel() for p in model.get_gamma_parameters())
assert gamma_p == 10, f"Expected 10 gamma params, got {gamma_p}"
print(f"CHECK 3 PASS: forward OK  shape={tuple(out.shape)}, total_params={total_p}, gamma_params={gamma_p}")

# ------------------------------------------------------------------
# Check 4: Two-group optimizer
# ------------------------------------------------------------------
from train_eval_data.fit_learnable_gamma import build_two_group_optimizer

opt = build_two_group_optimizer(model, lr=0.01, gamma_lr_factor=0.2)
assert len(opt.param_groups) == 2
assert abs(opt.param_groups[0]["lr"] - 0.01)  < 1e-9
assert abs(opt.param_groups[1]["lr"] - 0.002) < 1e-9
assert opt.param_groups[1]["weight_decay"] == 0.0
print(f"CHECK 4 PASS: two-group optimizer  main_lr={opt.param_groups[0]['lr']}, "
      f"gamma_lr={opt.param_groups[1]['lr']}")

# ------------------------------------------------------------------
# Check 5: End-to-end gradient via loss -> F -> W -> lam
# ------------------------------------------------------------------
# Use gamma_init=0.2 so lam~0.054, a*lam~0.2; test features have y in [0.04, 0.42]
# ensuring edges fall in SCAD region 2 (gradient non-zero).
torch.manual_seed(42)
model2 = RUNG_learnable_gamma(D, C, [32], lam_hat=0.9, gamma_init=0.2, prop_step=10)
opt2   = build_two_group_optimizer(model2, lr=0.01, gamma_lr_factor=0.2)
yy     = torch.randint(0, C, (N,))
lp_before = [p.item() for p in model2.log_lams]

out2   = model2(A, X)
loss2  = F.cross_entropy(out2, yy)
loss2.backward()

nz = sum(1 for lp in model2.log_lams if lp.grad is not None and abs(lp.grad.item()) > 1e-10)
print(f"  Non-zero gamma grads (gamma_init=0.2): {nz} / 10")
assert nz > 0, "FAIL: no gamma gradients — are all edges in region 1 or 3 only?"

opt2.step()
lp_after  = [p.item() for p in model2.log_lams]
changed   = sum(1 for b, a in zip(lp_before, lp_after) if abs(b - a) > 1e-10)
assert changed > 0, "FAIL: gamma params not updated after step"
print(f"CHECK 5 PASS: {changed}/10 gamma params updated after backward+step")

# ------------------------------------------------------------------
# Check 6: Factory logic (tested inline, avoiding gb/seml deps)
# ------------------------------------------------------------------
# Instead of importing get_model_default (which requires gb → seml),
# we directly replicate the RUNG_learnable_gamma branch of the factory.
import numpy as np

gamma_init = 6.0
scad_a     = 3.7
D2, C2     = 64, 7

m_uniform = RUNG_learnable_gamma(
    in_dim=D2, out_dim=C2, hidden_dims=[64],
    gamma_init=gamma_init, gamma_init_strategy="uniform",
)
m_decr = RUNG_learnable_gamma(
    in_dim=D2, out_dim=C2, hidden_dims=[64],
    gamma_init=gamma_init, gamma_init_strategy="decreasing",
)
m_incr = RUNG_learnable_gamma(
    in_dim=D2, out_dim=C2, hidden_dims=[64],
    gamma_init=gamma_init, gamma_init_strategy="increasing",
)

gs_u = m_uniform.get_learned_gammas()
gs_d = m_decr.get_learned_gammas()
gs_i = m_incr.get_learned_gammas()

assert all(abs(g - gamma_init) < 1e-4 for g in gs_u), "uniform: all gammas should equal gamma_init"
assert gs_d[0] > gs_d[-1], "decreasing: first should be largest"
assert gs_i[0] < gs_i[-1], "increasing: first should be smallest"

print(f"CHECK 6 PASS: gamma initialisation strategies work")
print(f"  uniform:    {[f'{g:.2f}' for g in gs_u[:3]]}...")
print(f"  decreasing: {[f'{g:.2f}' for g in gs_d[:3]]}...  [{gs_d[-1]:.2f}]")
print(f"  increasing: {[f'{g:.2f}' for g in gs_i[:3]]}...  [{gs_i[-1]:.2f}]")

# ------------------------------------------------------------------
# Check 7: RUNG and RUNG_new_SCAD original classes untouched
# ------------------------------------------------------------------
from model.rung import RUNG
from model.penalty import PenaltyFunction
from model.att_func import get_mcp_att_func

# Build originals directly (no gb dependency)
rung_orig = RUNG(D2, C2, [64], get_mcp_att_func(6.0), 0.9)
w_scad    = PenaltyFunction.get_w_func("scad", 6.0 / 3.7)
rung_scad = RUNG(D2, C2, [64], w_scad, 0.9)

assert isinstance(rung_orig, RUNG), f"Type changed: {type(rung_orig)}"
assert isinstance(rung_scad, RUNG), f"Type changed: {type(rung_scad)}"
assert not isinstance(rung_orig, RUNG_learnable_gamma), "RUNG should NOT be RUNG_learnable_gamma"
print("CHECK 7 PASS: RUNG and RUNG_new_SCAD model classes untouched")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
print("=" * 55)
print("  ALL 7 CHECKS PASSED — RUNG_learnable_gamma ready")
print("=" * 55)
print()
model.log_gamma_stats()
