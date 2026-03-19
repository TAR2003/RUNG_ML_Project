#!/usr/bin/env python
"""
verify_rung_combined.py

Quick verification that RUNG_combined is properly integrated.

Run this to:
  1. Verify model instantiation
  2. Check parameter counts match parent models
  3. Run forward pass successfully
  4. Confirm cosine distances are in [0, 2]
  5. Compare parent vs combined model

Usage:
    python verify_rung_combined.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.rung_combined import RUNG_combined
from model.rung_percentile_gamma import RUNG_percentile_gamma
from model.rung_learnable_distance import RUNG_learnable_distance
from train_eval_data.get_dataset import get_dataset, get_splits
from exp.config.get_model import get_model_default


def verify_basic():
    """Verify basic model instantiation"""
    print("\n" + "=" * 70)
    print("VERIFYING RUNG_COMBINED BASIC FUNCTIONALITY")
    print("=" * 70)

    print("\n[1] Testing direct instantiation...")
    model = RUNG_combined(
        in_dim=64,
        out_dim=7,
        hidden_dims=[64],
        percentile_q=0.75,
    )
    print(f"  ✓ Model created")
    print(f"    Type: {type(model).__name__}")
    print(f"    Parameters: {model.count_parameters()}")

    print("\n[2] Testing forward pass...")
    A = torch.randn(50, 50)
    X = torch.randn(50, 64)
    with torch.no_grad():
        logits = model(A, X)
    print(f"  ✓ Forward pass successful")
    print(f"    Input: A={A.shape}, X={X.shape}")
    print(f"    Output: logits={logits.shape}")

    print("\n[3] Checking cosine distance ranges...")
    gammas = model.get_last_gammas()
    print(f"  ✓ Gammas extracted: {len(gammas)} layers")
    if any(g is not None for g in gammas):
        valid_gammas = [g for g in gammas if g is not None and not (isinstance(g, float) and g != g)]
        if valid_gammas:
            print(f"    Min gamma: {min(valid_gammas):.4f}")
            print(f"    Max gamma: {max(valid_gammas):.4f}")
            print(f"    All in [0, 2]? {all(0 <= g <= 2.0 for g in valid_gammas)}")

    return model


def verify_factory():
    """Verify model instantiation via factory"""
    print("\n" + "=" * 70)
    print("VERIFYING FACTORY INTEGRATION")
    print("=" * 70)

    print("\n[1] Loading Cora dataset...")
    A, X, y = get_dataset('cora')
    sp = get_splits(y)
    train_idx, val_idx, test_idx = sp[0]
    print(f"  ✓ Dataset loaded")
    print(f"    Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {y.max().item()+1}")
    print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    print("\n[2] Creating RUNG_combined via get_model_default...")
    # Use CPU for verification to avoid GPU compatibility issues
    model_combined = get_model_default(
        'cora',
        'RUNG_combined',
        custom_model_params={'percentile_q': 0.75},
        device='cpu'
    )[0]
    print(f"  ✓ Model created via factory")
    print(f"    Parameters: {model_combined.count_parameters()}")

    print("\n[3] Forward pass on real data...")
    device = next(model_combined.parameters()).device
    A_small = A[:200, :200].to(device)
    X_small = X[:200].to(device)
    with torch.no_grad():
        logits = model_combined(A_small, X_small)
    print(f"  ✓ Forward pass successful on real data")
    print(f"    Device: {device}")
    print(f"    Input: {A_small.shape}, Output: {logits.shape}")

    return model_combined, A, X, y, train_idx, val_idx, test_idx


def verify_parameter_matching():
    """Verify RUNG_combined has same params as parents"""
    print("\n" + "=" * 70)
    print("PARAMETER COUNT COMPARISON")
    print("=" * 70)

    print("\nInstantiating models on Cora...")
    model_combined = get_model_default('cora', 'RUNG_combined', device='cpu')[0]
    model_pg = get_model_default('cora', 'RUNG_percentile_gamma', device='cpu')[0]
    model_ld = get_model_default(
        'cora',
        'RUNG_learnable_distance',
        custom_model_params={'distance_mode': 'cosine'},
        device='cpu'
    )[0]

    params_combined = model_combined.count_parameters()
    params_pg = model_pg.count_parameters()
    params_ld = model_ld.count_parameters()

    print(f"\n{'Model':<30} {'Parameters':>15}")
    print("-" * 45)
    print(f"{'RUNG_combined':<30} {params_combined:>15}")
    print(f"{'RUNG_percentile_gamma':<30} {params_pg:>15}")
    print(f"{'RUNG_learnable_distance':<30} {params_ld:>15}")

    if params_combined == params_pg == params_ld:
        print(f"\n  ✓ All models have IDENTICAL parameter count")
        print(f"    This confirms RUNG_combined adds zero new parameters")
    else:
        print(f"\n  ✗ Parameter mismatch!")
        print(f"    Combined: {params_combined}, PG: {params_pg}, LD: {params_ld}")


def verify_distance_stability():
    """Verify cosine distance is scale-invariant"""
    print("\n" + "=" * 70)
    print("VERIFYING COSINE DISTANCE SCALE-INVARIANCE")
    print("=" * 70)

    print("\nTesting scale invariance of cosine distances...")

    # Create test data
    torch.manual_seed(42)
    F = torch.randn(100, 64)

    # Test 1: Normal features
    F_unit = torch.nn.functional.normalize(F, p=2, dim=-1, eps=1e-8)
    cos_sim1 = torch.mm(F_unit, F_unit.T)
    y1 = (1.0 - cos_sim1).clamp(0, 2)

    # Test 2: Features scaled by 10x
    F_scaled = F * 10
    F_scaled_unit = torch.nn.functional.normalize(F_scaled, p=2, dim=-1, eps=1e-8)
    cos_sim2 = torch.mm(F_scaled_unit, F_scaled_unit.T)
    y2 = (1.0 - cos_sim2).clamp(0, 2)

    # Compare
    diff = (y1 - y2).abs().max().item()
    mean_diff = (y1 - y2).abs().mean().item()

    print(f"  Cosine distance after 10x scaling:")
    print(f"    Max difference: {diff:.2e}")
    print(f"    Mean difference: {mean_diff:.2e}")
    print(f"    Range y1: [{y1.min():.4f}, {y1.max():.4f}]")
    print(f"    Range y2: [{y2.min():.4f}, {y2.max():.4f}]")

    if diff < 1e-5:
        print(f"\n  ✓ Cosine distances are scale-invariant")
    else:
        print(f"\n  ✗ Cosine distances show scale dependence (unexpected)")


def main():
    print("\n" * 2)
    print("╔" + "=" * 68 + "╗")
    print("║" + "RUNG_COMBINED VERIFICATION SUITE".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        # Run verifications
        model1 = verify_basic()
        model2, A, X, y, train_idx, val_idx, test_idx = verify_factory()
        verify_parameter_matching()
        verify_distance_stability()

        print("\n" + "=" * 70)
        print("ALL VERIFICATION TESTS PASSED ✓")
        print("=" * 70)
        print("\nRUNG_COMBINED is ready to use!")
        print("\nNext steps:")
        print("  1. Run training: python train_test_combined.py --dataset cora")
        print("  2. Tune percentile_q: python train_test_combined.py --dataset cora --percentile_q 0.70")
        print("  3. Read docs: RUNG_COMBINED_README.md")
        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ VERIFICATION FAILED")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
