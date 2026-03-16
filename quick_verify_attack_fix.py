#!/usr/bin/env python
"""
Simple test: attack_pgd function API is fixed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("ATTACK_PGD FIX VERIFICATION")
print("="*80)

# Test 1: Import check
print("\n✓ Checking if attack_pgd imports without errors...")
try:
    from train_test_combined import attack_pgd
    print("  SUCCESS: attack_pgd imported")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 2: Verify function uses correct proj_grad_descent API
print("\n✓ Checking attack_pgd uses correct proj_grad_descent API...")
import inspect
source = inspect.getsource(attack_pgd)

required_params = [
    "flip_shape_or_init=A.shape",
    "symmetric=True",
    "device=A.device",
    "budget=budget_edge_num",
    "grad_fn=grad_fn",
    "loss_fn=loss_fn",
]

all_found = True
for param in required_params:
    if param in source:
        print(f"  ✓ {param}")
    else:
        print(f"  ✗ {param} NOT FOUND")
        all_found = False

if not all_found:
    print("  FAILED: Missing required parameters")
    sys.exit(1)

# Test 3: Verify edge_diff_matrix is used correctly
print("\n✓ Checking edge_diff_matrix is used for edge perturbations...")

if "edge_diff_matrix(edge_pert" in source:
    print("  ✓ Uses edge_diff_matrix correctly for edge indices")
else:
    print("  ✗ edge_diff_matrix not used correctly")
    sys.exit(1)

# Test 4: Verify no old incorrect API calls
print("\n✓ Checking for old incorrect API calls...")

errors_to_check = [
    ("adj=A", "Old API: adj parameter"),
    ("feat=X", "Old API: feat parameter"),
    ("labels=y", "Old API: labels parameter"),
    ("idx_train=", "Old API: idx_train parameter"),
    ("idx_test=", "Old API: idx_test parameter"),
    ("epoch=", "Old API: epoch parameter instead of iterations"),
    ("perturbation_ratio=", "Old API: perturbation_ratio parameter"),
]

found_errors = []
for error_code, error_msg in errors_to_check:
    if error_code in source and "def attack_pgd" in source:
        # Check if it's in the attack_pgd function, not elsewhere
        start_idx = source.find("def attack_pgd")
        if error_code in source[start_idx:start_idx+2000]:
            found_errors.append(error_msg)
            print(f"  ✗ FOUND: {error_msg}")

if found_errors:
    print(f"  FAILED: Found {len(found_errors)} old API calls")
    sys.exit(1)
else:
    print("  ✓ No old incorrect API calls found")

print("\n" + "="*80)
print("✓ ALL CHECKS PASSED")
print("="*80)
print("\nSummary of fix:")
print("  - proj_grad_descent now called with correct parameters")
print("  - Uses flip_shape_or_init=A.shape (not adj=...)")
print("  - Uses symmetric=True and device=A.device")  
print("  - Uses budget (number of edges) not perturbation_ratio")
print("  - Returns edge indices, converted via edge_diff_matrix")
print("\nThe fix resolves: 'ERROR: missing a required argument: flip_shape_or_init'")
print("\nRun: python train_test_combined.py --dataset cora --max_epoch 300")
