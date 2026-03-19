#!/usr/bin/env python
"""
diagnose_attack_strength.py

Check whether training-time attacks are as strong as test-time attacks.

PROBLEM:
--------
If training uses weak PGD attacks (e.g., 20 steps) but testing uses strong
attacks (e.g., 200 steps), the model trains to resist weak attacks and
fails against strong ones. This is a hidden vulnerability.

THRESHOLD:
---------
If accuracy gap between 20-step and 200-step attack is > 5%, then the
model is vulnerable to stronger attacks seen at test time.

THEORY:
--------
A well-trained robust model should have SMALL accuracy differences
across different attack strengths:
   - acc(0-step):   ~80% (clean)
   - acc(20-step):  ~75% (train attack strength)
   - acc(100-step): ~74% (medium strength)
   - acc(200-step): ~73% (test attack strength)

If gaps are large:
   - acc(20-step):  ~75%
   - acc(200-step): ~60%  ← LARGE 15% gap = model not robust

USAGE:
    python diagnose_attack_strength.py --model RUNG_percentile_gamma \
                                       --dataset cora
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_eval_data.get_dataset import get_dataset, get_splits
from exp.config.get_model import get_model_default
from experiments.run_ablation import pgd_attack
from utils import accuracy


def diagnose_attack_strength(
    model_name: str = 'RUNG_percentile_gamma',
    dataset: str = 'cora',
    device: str = 'cpu',
    train_steps: int = 20,
    medium_steps: int = 100,
    test_steps: int = 200,
    budget: float = 0.20,
) -> dict:
    """
    Check if training attack is as strong as test attack.

    Args:
        model_name: Model architecture name
        dataset: Dataset name
        device: 'cpu' or 'cuda'
        train_steps: Attack steps used during training
        medium_steps: Medium-strength attack 
        test_steps: Attack steps used at test time
        budget: Attack budget

    Returns:
        Dict with accuracies at each strength level
    """

    print(f"\n{'='*70}")
    print(f"{'DIAGNOSTIC 2: Training attack strength vs test attack':^70}")
    print(f"{'='*70}\n")

    # Load data and train model
    print("Loading data...")
    A, X, y = get_dataset(dataset)
    sp = get_splits(y)
    train_idx, val_idx, test_idx = sp[0]

    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    test_idx = test_idx.to(device)

    n_edges = A.count_nonzero().item() // 2
    budget_edge_num = max(1, int(budget * n_edges))

    print(f"  Dataset: {dataset}")
    print(f"  Budget: {budget:.0%} = {budget_edge_num} edges\n")

    # Build and train model
    print(f"Building and training {model_name}...")
    model, _ = get_model_default(dataset, model_name)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    for epoch in range(1, 151):
        model.train()
        optimizer.zero_grad()
        logits = model(A, X)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")

    # Evaluate on clean graph
    model.eval()
    with torch.no_grad():
        clean_acc = accuracy(model(A, X)[test_idx], y[test_idx])
    print(f"\nClean test accuracy: {clean_acc:.4f}\n")

    # Attack with different step counts
    results = {}
    step_configs = [
        ('train (20 steps)', train_steps),
        ('medium (100 steps)', medium_steps),
        ('test (200 steps)', test_steps),
    ]

    for label, n_steps in step_configs:
        print(f"Attacking with {label}...", end='', flush=True)
        model.eval()
        attacked = pgd_attack(
            model, A, X, y, test_idx, budget_edge_num, iterations=n_steps
        )

        with torch.no_grad():
            acc = accuracy(model(attacked, X)[test_idx], y[test_idx])
        results[n_steps] = acc
        print(f" accuracy={acc:.4f}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    gap_20_200 = (results[train_steps] - results[test_steps]).abs()

    print(f"  Clean accuracy:              {clean_acc:.4f}")
    print(f"  Accuracy under   20-step:    {results[train_steps]:.4f}")
    print(f"  Accuracy under  100-step:    {results[medium_steps]:.4f}")
    print(f"  Accuracy under  200-step:    {results[test_steps]:.4f}")
    print(f"\n  Gap (20-step vs 200-step):   {gap_20_200:.4f}")

    print(f"\n{'='*70}")

    if gap_20_200 > 0.05:
        print(f"{'DIAGNOSIS: Training attack too weak':^70}")
        print(f"{'='*70}\n")
        print(f"❌ Large accuracy gap: {gap_20_200:.1%}")
        print(f"   Training uses {train_steps}-step attacks")
        print(f"   Testing uses {test_steps}-step attacks")
        print(f"   Model trained to resist weak attacks, vulnerable to strong ones\n")
        print("FIX REQUIRED:")
        print(f"  Increase training attack steps from {train_steps} to {test_steps}")
        print(f"  This will slow training by ~{test_steps/train_steps:.0f}x but provide true robustness")
        print(f"  Alternatively, use {medium_steps} steps as compromise (3x slower, better robustness)\n")
        return results

    elif gap_20_200 > 0.02:
        print(f"{'DIAGNOSIS: Small attack strength gap':^70}")
        print(f"{'='*70}\n")
        print(f"⚠️  Moderate gap: {gap_20_200:.1%}")
        print(f"   Consider increasing training attack strength for better robustness\n")
        return results

    else:
        print(f"{'DIAGNOSIS: Attack strength matched ✓':^70}")
        print(f"{'='*70}\n")
        print(f"✅ Small gap: {gap_20_200:.1%}")
        print(f"   Training and test attacks are well-matched")
        print(f"   Attack strength NOT the failure cause\n")
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Diagnose attack strength mismatch'
    )
    parser.add_argument(
        '--model', default='RUNG_percentile_gamma',
        help='Model name'
    )
    parser.add_argument('--dataset', default='cora', help='Dataset name')
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )
    parser.add_argument(
        '--train-steps', type=int, default=20,
        help='Training attack steps'
    )
    parser.add_argument(
        '--medium-steps', type=int, default=100,
        help='Medium attack steps'
    )
    parser.add_argument(
        '--test-steps', type=int, default=200,
        help='Test attack steps'
    )
    parser.add_argument('--budget', type=float, default=0.20, help='Attack budget')

    args = parser.parse_args()

    results = diagnose_attack_strength(
        model_name=args.model,
        dataset=args.dataset,
        device=args.device,
        train_steps=args.train_steps,
        medium_steps=args.medium_steps,
        test_steps=args.test_steps,
        budget=args.budget,
    )
