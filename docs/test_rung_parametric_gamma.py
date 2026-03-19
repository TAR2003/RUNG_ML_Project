"""Verification checks for RUNG_parametric_gamma — run from project root."""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

print("\n" + "="*70)
print("VERIFICATION TESTS: RUNG_parametric_gamma")
print("="*70)

# ------------------------------------------------------------------
# Check 1: Import
# ------------------------------------------------------------------
try:
    from model.rung_parametric_gamma import RUNG_parametric_gamma
    from model.rung_learnable_gamma import RUNG_learnable_gamma
    print("✓ CHECK 1 PASS: model import OK")
except ImportError as e:
    print(f"✗ CHECK 1 FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Check 2: Instantiation
# ------------------------------------------------------------------
try:
    torch.manual_seed(42)
    model = RUNG_parametric_gamma(
        in_dim=64, out_dim=7, hidden_dims=[64],
        lam_hat=0.9, gamma_0_init=3.0, decay_rate_init=0.85,
        prop_step=10, dropout=0.5
    )
    print(f"✓ CHECK 2 PASS: model instantiation OK")
    print(f"  - gamma_0 = {model.get_gamma_0_value():.4f}")
    print(f"  - decay_rate = {model.get_decay_rate_value():.4f}")
except Exception as e:
    print(f"✗ CHECK 2 FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Check 3: Parameter count — should have only 2 extra params vs MLP
# ------------------------------------------------------------------
try:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gamma_params = sum(p.numel() for p in model.get_gamma_parameters() if p.requires_grad)
    
    assert gamma_params == 2, f"Expected 2 gamma params, got {gamma_params}"
    assert total_params > 2, f"Expected total_params > 2, got {total_params}"
    
    print(f"✓ CHECK 3 PASS: parameter count OK")
    print(f"  - gamma schedule params: {gamma_params} (log_gamma_0, raw_decay)")
    print(f"  - total params: {total_params}")
except AssertionError as e:
    print(f"✗ CHECK 3 FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Check 4: Gradient flow to both schedule parameters
# ------------------------------------------------------------------
try:
    torch.manual_seed(42)
    # Use gamma_0_init=0.2 so lam~0.054, a*lam~0.2; test features have y in [0.04, 0.42]
    # ensuring edges fall in SCAD region 2 (gradient non-zero).
    model2 = RUNG_parametric_gamma(64, 7, [64], gamma_0_init=0.2, prop_step=5)
    N, E = 100, 300
    A = (torch.rand(N, N) > 0.8).float()
    A = A + A.t()
    A.fill_diagonal_(0)
    X = torch.randn(N, 64)
    y = torch.randint(0, 7, (N,))
    
    logits = model2(A, X)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    
    grad_g0 = model2.log_gamma_0.grad
    grad_decay = model2.raw_decay.grad
    
    assert grad_g0 is not None, "log_gamma_0.grad is None"
    assert grad_decay is not None, "raw_decay.grad is None"
    assert grad_g0.abs() > 1e-10, f"log_gamma_0.grad too small: {grad_g0.item()}"
    assert grad_decay.abs() > 1e-10, f"raw_decay.grad too small: {grad_decay.item()}"
    
    print(f"✓ CHECK 4 PASS: gradient flow OK")
    print(f"  - log_gamma_0.grad = {grad_g0.item():.6f}")
    print(f"  - raw_decay.grad = {grad_decay.item():.6f}")
except AssertionError as e:
    print(f"✗ CHECK 4 FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Check 5: Forward pass shape
# ------------------------------------------------------------------
try:
    torch.manual_seed(42)
    N, D, C = 50, 16, 7
    A = (torch.rand(N, N) > 0.8).float()
    A = A + A.t()
    A.fill_diagonal_(0)
    X = torch.randn(N, D)
    
    model3 = RUNG_parametric_gamma(D, C, [32], prop_step=10)
    out = model3(A, X)
    
    assert out.shape == (N, C), f"Expected shape ({N}, {C}), got {out.shape}"
    print(f"✓ CHECK 5 PASS: forward pass shape OK")
    print(f"  - input X: {X.shape}")
    print(f"  - output logits: {out.shape}")
except AssertionError as e:
    print(f"✗ CHECK 5 FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Check 6: Gamma schedule is decreasing
# ------------------------------------------------------------------
try:
    torch.manual_seed(42)
    model4 = RUNG_parametric_gamma(64, 7, [64], prop_step=10)
    N, E = 100, 300
    A = (torch.rand(N, N) > 0.8).float()
    A = A + A.t()
    A.fill_diagonal_(0)
    X = torch.randn(N, 64)
    
    with torch.no_grad():
        _ = model4(A, X)
    
    gammas = model4.get_learned_gammas()
    
    # Check that gammas are decreasing
    for k in range(len(gammas) - 1):
        assert gammas[k] >= gammas[k+1], \
            f"Gamma not decreasing at layer {k}: {gammas[k]:.4f} < {gammas[k+1]:.4f}"
    
    print(f"✓ CHECK 6 PASS: gamma schedule decreasing OK")
    print(f"  - gammas = [{', '.join(f'{g:.4f}' for g in gammas)}]")
except AssertionError as e:
    print(f"✗ CHECK 6 FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Check 7: Fewer parameters than learnable_gamma
# ------------------------------------------------------------------
try:
    torch.manual_seed(42)
    model_lg = RUNG_learnable_gamma(64, 7, [64], prop_step=10)
    model_pg = RUNG_parametric_gamma(64, 7, [64], prop_step=10)
    
    params_lg = sum(p.numel() for p in model_lg.parameters() if p.requires_grad)
    params_pg = sum(p.numel() for p in model_pg.parameters() if p.requires_grad)
    
    assert params_pg < params_lg, \
        f"parametric_gamma has more params ({params_pg}) than learnable_gamma ({params_lg})"
    
    diff = params_lg - params_pg
    print(f"✓ CHECK 7 PASS: fewer parameters than learnable_gamma")
    print(f"  - learnable_gamma params: {params_lg}")
    print(f"  - parametric_gamma params: {params_pg}")
    print(f"  - difference: {diff} parameters saved")
except AssertionError as e:
    print(f"✗ CHECK 7 FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Check 8: Log stats formatting
# ------------------------------------------------------------------
try:
    torch.manual_seed(42)
    model5 = RUNG_parametric_gamma(64, 7, [64], prop_step=5)
    N, E = 50, 200
    A = (torch.rand(N, N) > 0.8).float()
    A = A + A.t()
    A.fill_diagonal_(0)
    X = torch.randn(N, 64)
    
    with torch.no_grad():
        _ = model5(A, X)
    
    # Run log_stats (should not raise)
    model5.log_gamma_stats()
    print(f"✓ CHECK 8 PASS: log_gamma_stats formatting OK")
except Exception as e:
    print(f"✗ CHECK 8 FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Check 9: get_aggregated_features
# ------------------------------------------------------------------
try:
    torch.manual_seed(42)
    model6 = RUNG_parametric_gamma(64, 7, [64], prop_step=5)
    N, D, C = 50, 64, 7
    A = (torch.rand(N, N) > 0.8).float()
    A = A + A.t()
    A.fill_diagonal_(0)
    X = torch.randn(N, D)
    
    features = model6.get_aggregated_features(A, X)
    assert features.shape == (N, C), f"Expected shape ({N}, {C}), got {features.shape}"
    
    print(f"✓ CHECK 9 PASS: get_aggregated_features OK")
    print(f"  - output shape: {features.shape}")
except AssertionError as e:
    print(f"✗ CHECK 9 FAIL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Check 10: Initialization consistency
# ------------------------------------------------------------------
try:
    gamma_0_init = 2.5
    decay_rate_init = 0.80
    
    model7 = RUNG_parametric_gamma(
        64, 7, [64], 
        gamma_0_init=gamma_0_init,
        decay_rate_init=decay_rate_init,
        prop_step=3
    )
    
    # Check initial values are close to specified
    g0_actual = model7.get_gamma_0_value()
    r_actual = model7.get_decay_rate_value()
    
    assert abs(g0_actual - gamma_0_init) < 0.01, \
        f"gamma_0 init mismatch: {g0_actual} vs {gamma_0_init}"
    assert abs(r_actual - decay_rate_init) < 0.01, \
        f"decay_rate init mismatch: {r_actual} vs {decay_rate_init}"
    
    print(f"✓ CHECK 10 PASS: initialization consistency OK")
    print(f"  - gamma_0: {g0_actual:.4f} (target: {gamma_0_init})")
    print(f"  - decay_rate: {r_actual:.4f} (target: {decay_rate_init})")
except AssertionError as e:
    print(f"✗ CHECK 10 FAIL: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("ALL CHECKS PASSED ✓")
print("="*70 + "\n")
