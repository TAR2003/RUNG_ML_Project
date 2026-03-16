#!/usr/bin/env python
"""
diagnose_attack.py

Check whether the PGD attack is TRULY adaptive.

An adaptive attack should produce DIFFERENT edge perturbations when the model
weights change. If the same edges are perturbed regardless of model state,
the attack is not computing gradients through the model correctly.

THEORY:
--------
Attack is ADAPTIVE if:  overlap_fraction < 0.90
Attack is NOT ADAPTIVE if:  overlap_fraction > 0.90

IMPORTANT: This diagnostic must run BEFORE adversarial training.
If attack is not adaptive, adversarial training will not work no matter
how long you train or how strong your curriculum is.

USAGE:
    python diagnose_attack.py --dataset cora --budget 0.10 --steps 20
"""

import torch
import torch.nn.functional as F
import argparse
import copy
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_eval_data.get_dataset import get_dataset, get_splits
from exp.config.get_model import get_model_default
from experiments.run_ablation import pgd_attack


def diagnose_attack_adaptivity(
    dataset: str = 'cora',
    budget: float = 0.10,
    n_steps: int = 20,
    device: str = 'cpu',
) -> bool:
    """
    Check if attack is adaptive by comparing attacks on different model states.

    Args:
        dataset: Name of dataset
        budget: Attack budget as fraction (0.10 = 10%)
        n_steps: PGD iterations  
        device: 'cpu' or 'cuda'

    Returns:
        True if adaptive, False if not adaptive
    """

    print(f"\n{'='*70}")
    print(f"{'DIAGNOSTIC 1: Is the PGD attack truly adaptive?':^70}")
    print(f"{'='*70}\n")

    # Load data
    print("Loading data...")
    A, X, y = get_dataset(dataset)
    sp = get_splits(y)
    train_idx, val_idx, test_idx = sp[0]

    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    test_idx = test_idx.to(device)

    n_nodes = A.shape[0]
    n_edges = A.count_nonzero().item() // 2
    budget_edge_num = max(1, int(budget * n_edges))

    print(f"  Dataset: {dataset}")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {n_edges}")
    print(f"  Budget: {budget:.0%} = {budget_edge_num} edges")
    print(f"  PGD steps: {n_steps}\n")

    # Build initial model
    print("Building RUNG_percentile_gamma model...")
    model_A, _ = get_model_default(dataset, 'RUNG_percentile_gamma')
    model_A = model_A.to(device)
    print(f"  Model: {model_A.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model_A.parameters())}\n")

    # Attack with untrained model_A
    print("Step 1: Attack on UNTRAINED model...")
    model_A.eval()
    attacked_A = pgd_attack(
        model_A, A, X, y, test_idx, budget_edge_num, iterations=n_steps
    )
    edges_A = set(map(tuple, torch.stack(torch.where(attacked_A > 0.5)).T.tolist()))
    original_edges = set(map(tuple, torch.stack(torch.where(A > 0.5)).T.tolist()))
    new_edges_A = edges_A - original_edges
    print(f"  New edges added: {len(new_edges_A)}\n")

    # Train model_A for 50 steps to change weights significantly
    print("Step 2: Train model briefly to change weights...")
    model_A.train()
    optimizer_A = torch.optim.Adam(model_A.parameters(), lr=0.01)
    for step in range(50):
        optimizer_A.zero_grad()
        logits = model_A(A, X)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer_A.step()
    print(f"  Training complete. Loss decreased from ~1.0 to {loss.item():.4f}\n")

    # Attack with trained model_A
    print("Step 3: Attack on TRAINED model...")
    model_A.eval()
    attacked_A2 = pgd_attack(
        model_A, A, X, y, test_idx, budget_edge_num, iterations=n_steps
    )
    edges_A2 = set(map(tuple, torch.stack(torch.where(attacked_A2 > 0.5)).T.tolist()))
    new_edges_A2 = edges_A2 - original_edges
    print(f"  New edges added: {len(new_edges_A2)}\n")

    # Compare
    print("Step 4: Analyzing overlap...\n")
    common = new_edges_A & new_edges_A2
    only_in_first = new_edges_A - new_edges_A2
    only_in_second = new_edges_A2 - new_edges_A

    overlap_frac = len(common) / max(len(new_edges_A), 1)
    diff_frac = 1.0 - overlap_frac

    print(f"  New edges (untrained): {len(new_edges_A)}")
    print(f"  New edges (trained):   {len(new_edges_A2)}")
    print(f"  Common edges:          {len(common)}")
    print(f"  Only in first:         {len(only_in_first)}")
    print(f"  Only in second:        {len(only_in_second)}")
    print(f"\n  Overlap fraction: {overlap_frac:.1%}")
    print(f"  Difference fraction: {diff_frac:.1%}\n")

    # Diagnosis
    print(f"{'='*70}")
    if overlap_frac > 0.85:
        print(f"{'DIAGNOSIS: ATTACK IS NOT ADAPTIVE':^70}")
        print(f"{'='*70}\n")
        print("❌ The attack adds the SAME edges regardless of model state.")
        print(f"   Overlap: {overlap_frac:.1%} (threshold: 85%)")
        print("\nWHAT THIS MEANS:")
        print("  - Attack is NOT using model gradients correctly")
        print("  - Attack is likely using cached or detached model")
        print("  - Adversarial training will NOT improve robustness")
        print("\nFIX REQUIRED:")
        print("  Check experiments/run_ablation.py::pgd_attack()")
        print("  Ensure:")
        print("    1. model.eval() is called BEFORE grad computation")
        print("    2. loss_fn() uses FRESH forward pass (no detach)")
        print("    3. torch.autograd.grad(loss, flip) computes wrt attack params")
        print("    4. Model is in eval mode during attack (no dropout noise)")
        print()
        return False
    else:
        print(f"{'DIAGNOSIS: ATTACK IS ADAPTIVE ✓':^70}")
        print(f"{'='*70}\n")
        print(f"✅ The attack produces DIFFERENT edges with different models.")
        print(f"   Overlap: {overlap_frac:.1%} (acceptable, <85%)")
        print("\nWHAT THIS MEANS:")
        print("  - Attack uses gradients through model correctly")
        print("  - Attack adaptivity is NOT the failure cause")
        print("  - Root cause is elsewhere (see Diagnostic 2 and 3)")
        print()
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Diagnose PGD attack adaptivity'
    )
    parser.add_argument('--dataset', default='cora', help='Dataset name')
    parser.add_argument('--budget', type=float, default=0.10, help='Attack budget')
    parser.add_argument('--steps', type=int, default=20, help='PGD iterations')
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )

    args = parser.parse_args()

    is_adaptive = diagnose_attack_adaptivity(
        dataset=args.dataset,
        budget=args.budget,
        n_steps=args.steps,
        device=args.device,
    )

    exit(0 if is_adaptive else 1)
