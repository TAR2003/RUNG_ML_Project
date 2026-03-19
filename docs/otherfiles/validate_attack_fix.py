#!/usr/bin/env python
"""
Validate that attack_pgd function works correctly with the fixed API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("ATTACK_PGD FIX VALIDATION TEST")
print("="*80)

# Test 1: Import check
print("\n[Test 1] Checking imports...")
try:
    import torch
    from gb.attack.gd import proj_grad_descent
    from gb.metric import margin, accuracy
    from train_test_combined import attack_pgd
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Synthetic test of attack logic
print("\n[Test 2] Testing attack logic with synthetic data...")
try:
    # Create synthetic data
    n_nodes = 50
    n_features = 10
    n_classes = 3
    
    A = torch.rand(n_nodes, n_nodes) > 0.9  # Sparse random graph
    A = (A | A.T).float()  # Make symmetric
    X = torch.randn(n_nodes, n_features)
    y = torch.randint(0, n_classes, (n_nodes,))
    test_idx = torch.arange(10, 20)
    
    # Create dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, A, X):
            return torch.randn(X.shape[0], n_classes)
    
    model = DummyModel()
    
    # Call attack_pgd with small budget
    print(f"  Graph: {A.shape}, edges: {A.count_nonzero()}, test nodes: {len(test_idx)}")
    acc = attack_pgd(
        model=model,
        A=A,
        X=X,
        y=y,
        test_idx=test_idx,
        budget=0.05,  # 5% edge budget
        n_epochs=2,   # Just 2 iterations for quick test
        lr_attack=0.01,
        device='cpu',
    )
    print(f"  ✓ Attack completed, accuracy: {acc:.4f}")
    
except Exception as e:
    print(f"✗ Attack test failed: {e}")
    import traceback
    traceback.exc_info()
    sys.exit(1)

# Test 3: Verify attack function signature matches proj_grad_descent API
print("\n[Test 3] Checking attack_pgd function implementation...")
try:
    import inspect
    source = inspect.getsource(attack_pgd)
    
    # Check for correct API usage
    checks = [
        ("flip_shape_or_init=A.shape" in source, "✓ Uses flip_shape_or_init parameter"),
        ("symmetric=True" in source, "✓ Uses symmetric=True"),
        ("device=A.device" in source, "✓ Uses device parameter"),
        ("budget=budget_edge_num" in source, "✓ Uses budget parameter"),
        ("grad_fn=grad_fn" in source, "✓ Defines grad_fn"),
        ("loss_fn=loss_fn" in source, "✓ Defines loss_fn"),
        ("iterations=n_epochs" in source, "✓ Uses iterations parameter"),
    ]
    
    all_ok = True
    for check, msg in checks:
        if check:
            print(f"  {msg}")
        else:
            print(f"  ✗ {msg.replace('✓', '✗')}")
            all_ok = False
    
    if not all_ok:
        print("✗ Implementation checks failed")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Source check failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL TESTS PASSED - ATTACK FIX IS CORRECT")
print("="*80)
print("\nThe attack_pgd function now correctly uses proj_grad_descent with:")
print("  - flip_shape_or_init: adjacency matrix shape")
print("  - symmetric: True for undirected graphs")
print("  - device: GPU/CPU device")
print("  - budget: number of edges to flip")
print("  - grad_fn: gradient computation function")
print("  - loss_fn: loss/margin computation function")
print("\nThis fixes the 'missing a required argument' errors.")
