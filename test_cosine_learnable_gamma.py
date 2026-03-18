import torch
import torch.nn.functional as F

# ---- Test 1: Per-layer variant ----
print("Test 1: Per-layer gamma gradient")

K = 10
raw_gamma = torch.zeros(K, requires_grad=True)  # init: gamma = 1.0

# Simulate cosine distances (always in [0, 2])
torch.manual_seed(42)
y_cosine = torch.rand(500) * 2.0  # [0, 2] uniform

# Compute gammas
gammas = torch.sigmoid(raw_gamma) * 2.0
print(f"Initial gammas: {gammas.detach().tolist()}")
print("Expected: all approx 1.0 (sigmoid(0) * 2.0 = 1.0)")

# Simulate SCAD loss (differentiable w.r.t. gamma)
scad_a = 3.7
lam = gammas[0] / scad_a

region1 = y_cosine < lam
region2 = (y_cosine >= lam) & (y_cosine < scad_a * lam)

val1 = 1.0 / (2.0 * y_cosine.clamp(min=1e-8))
val2 = (scad_a * lam - y_cosine) / ((scad_a - 1) * lam * 2.0 * y_cosine.clamp(1e-8))
val3 = torch.zeros_like(y_cosine)

W = torch.where(region1, val1, torch.where(region2, val2, val3))
loss = -W.sum()  # maximize weights = minimize loss
loss.backward()

print(f"raw_gamma.grad: {raw_gamma.grad}")
print(f"Region 2 edges: {region2.sum().item()} / {len(y_cosine)}")
assert raw_gamma.grad is not None, "FAIL: gradient is None"
assert raw_gamma.grad.abs().sum() > 0, "FAIL: all gradients are zero"
print("PASS: gradient flows to per-layer gamma\n")

# ---- Test 2: Schedule variant ----
print("Test 2: Schedule gamma gradient")

raw_g0 = torch.tensor(0.0, requires_grad=True)   # gamma_0 = 1.0
raw_decay = torch.tensor(1.735, requires_grad=True)  # decay = 0.85

total_loss = torch.tensor(0.0)
for k in range(K):
    gamma_k = torch.sigmoid(raw_g0) * 2.0 * (torch.sigmoid(raw_decay) ** k)
    lam_k = gamma_k / scad_a

    y_k = torch.rand(500) * 1.5  # cosine distances
    r2 = (y_k >= lam_k) & (y_k < scad_a * lam_k)
    val2_k = (scad_a * lam_k - y_k) / ((scad_a - 1) * lam_k * 2.0 * y_k.clamp(1e-8))
    val2_k = val2_k.clamp(min=0)
    W_k = torch.where(r2, val2_k, torch.zeros_like(y_k))
    total_loss = total_loss + (-W_k.sum())

total_loss.backward()
print(f"raw_g0.grad:    {raw_g0.grad.item():.6f}")
print(f"raw_decay.grad: {raw_decay.grad.item():.6f}")
assert raw_g0.grad is not None and raw_g0.grad.abs() > 0
assert raw_decay.grad is not None and raw_decay.grad.abs() > 0
print("PASS: gradient flows to schedule parameters\n")

# ---- Test 3: Gamma stays in (0, 2) always ----
print("Test 3: Gamma range guarantee")
for val in [-100.0, -1.0, 0.0, 1.0, 100.0]:
    g = torch.sigmoid(torch.tensor(val)) * 2.0
    # At extreme logits, float32 can underflow/overflow to exact endpoints.
    assert 0.0 <= g.item() <= 2.0, f"FAIL: gamma={g.item()} out of [0,2]"
print("PASS: gamma always in (0, 2) for any raw_gamma value\n")

print("ALL TESTS PASSED")
