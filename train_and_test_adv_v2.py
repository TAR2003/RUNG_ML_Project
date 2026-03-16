#!/usr/bin/env python
"""
train_and_test_adv_v2.py

One-command training+attack script for RUNG_percentile_adv_v2.

Useful for full development iteration:
  1. Train RUNG_percentile_adv_v2 on clean graph
  2. Evaluate clean accuracy on test set
  3. Run PGD attacks at multiple budgets
  4. Report clean + attacked accuracy

This is the adversarial equivalent of train_test_combined.py.

USAGE:
    # Train and test v2 on Cora (default, 800 epochs, alpha=0.85, steps=100)
    python train_and_test_adv_v2.py --dataset cora

    # Custom parameters (faster iteration for development)
    python train_and_test_adv_v2.py --dataset cora \\
                                    --max_epoch 400 \\
                                    --train_pgd_steps 50 \\
                                    --warmup_epochs 50

    # Compare v1 vs v2
    python train_and_test_adv_v2.py --dataset cora  # trains v2
    python clean.py --model RUNG_percentile_adv_v1 --data cora  # trains v1
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import tqdm
from train_eval_data.get_dataset import get_dataset, get_splits
from utils import accuracy
from exp.config.get_model import get_model_default
from gb.attack.gd import proj_grad_descent
from gb.pert import edge_diff_matrix
from gb.metric import margin
from experiments.run_ablation import pgd_attack


# ============================================================
# CONSTANTS
# ============================================================

ATTACK_BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]


# ============================================================
# Training
# ============================================================

def train_percentile_adv_v2(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    max_epoch: int = 800,
    alpha: float = 0.85,
    train_pgd_steps: int = 100,
    warmup_epochs: int = 100,
    attack_freq: int = 3,
    lr: float = 0.05,
    weight_decay: float = 5e-4,
    patience: int = 150,
    device: str = 'cpu',
) -> dict:
    """
    Train RUNG_percentile_gamma using FIXED adversarial training (V2).

    Args:
        model:           RUNG_percentile_gamma instance
        A, X, y:         Graph data
        train_idx, val_idx, test_idx: Node indices
        max_epoch:       Maximum epochs (default 800)
        alpha:           Clean loss weight (default 0.85)
        train_pgd_steps: PGD steps during training (default 100)
        warmup_epochs:   Clean-only training epochs (default 100)
        attack_freq:     Regenerate attack every N epochs (default 3)
        lr:              Learning rate
        weight_decay:    L2 regularization
        patience:        Early stopping patience
        device:          torch device

    Returns:
        dict with training history and final accuracies
    """
    from train_eval_data.fit_percentile_adv_v2 import fit_percentile_adv_v2

    print(f"\n{'='*80}")
    print(f"{'RUNG_percentile_adv_v2: Train + Attack':^80}")
    print(f"{'='*80}\n")

    # Train
    best_val_acc, test_acc_clean = fit_percentile_adv_v2(
        model,
        A, X, y,
        train_idx, val_idx, test_idx,
        attack_fn=pgd_attack,
        alpha=alpha,
        train_pgd_steps=train_pgd_steps,
        max_epoch=max_epoch,
        warmup_epochs=warmup_epochs,
        attack_freq=attack_freq,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
    )

    results = {
        'best_val_acc': best_val_acc,
        'test_acc_clean': test_acc_clean,
    }

    return results


# ============================================================
# Attack
# ============================================================

def attack_pgd_v2(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    test_idx: torch.Tensor,
    budget: float,
    n_steps: int = 200,
    device: str = 'cpu',
) -> float:
    """
    Run PGD attack and return test accuracy under attack.

    Args:
        model:      Trained model
        A, X, y:    Graph data  
        test_idx:   Test indices
        budget:     Perturbation budget (0 to 1)
        n_steps:    Attack iterations
        device:     torch device

    Returns:
        Accuracy under attack
    """
    model = model.to(device)
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    test_idx = test_idx.to(device)

    # Prepare attack
    n_edges = A.count_nonzero().item() // 2
    budget_edge_num = max(1, int(budget * n_edges))

    # Run attack
    model.eval()
    A_attacked = pgd_attack(
        model, A, X, y, test_idx, budget_edge_num, iterations=n_steps
    )

    # Evaluate accuracy under attack
    with torch.no_grad():
        logits = model(A_attacked, X)
        attacked_acc = accuracy(logits[test_idx], y[test_idx])

    return attacked_acc.item()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train and test RUNG_percentile_adv_v2'
    )
    parser.add_argument(
        '--dataset', default='cora',
        help='Dataset name (default: cora)'
    )
    parser.add_argument(
        '--percentile_q', type=float, default=0.75,
        help='Percentile q for median aggregation (default: 0.75)'
    )
    parser.add_argument(
        '--use_layerwise_q', type=lambda x: x.lower() != 'false',
        default=False,
        help='Use different q for early/late layers (default: false)'
    )
    parser.add_argument(
        '--percentile_q_late', type=float, default=0.65,
        help='Late-layer percentile q (default: 0.65)'
    )
    # Training parameters
    parser.add_argument(
        '--max_epoch', type=int, default=800,
        help='Maximum training epochs (default: 800)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.05,
        help='Learning rate (default: 0.05)'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=5e-4,
        help='L2 regularization (default: 5e-4)'
    )
    # V2 adversarial parameters
    parser.add_argument(
        '--alpha', type=float, default=0.85,
        help='Clean loss weight (default: 0.85 = 85%% clean, 15%% adv)'
    )
    parser.add_argument(
        '--train_pgd_steps', type=int, default=100,
        help='PGD steps during training (default: 100)'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=100,
        help='Clean-only training epochs (default: 100)'
    )
    parser.add_argument(
        '--attack_freq', type=int, default=3,
        help='Regenerate attack every N epochs (default: 3)'
    )
    parser.add_argument(
        '--patience', type=int, default=150,
        help='Early stopping patience (default: 150)'
    )
    # Attack evaluation parameters
    parser.add_argument(
        '--attack_steps', type=int, default=200,
        help='PGD steps during test attacks (default: 200)'
    )
    parser.add_argument(
        '--skip_attack', action='store_true',
        help='Skip attack phase (training only)'
    )
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
        help='torch device (default: cuda if available, else cpu)'
    )

    args = parser.parse_args()

    # Create log directories
    os.makedirs('log', exist_ok=True)
    os.makedirs(f'log/{args.dataset}', exist_ok=True)
    os.makedirs(f'log/{args.dataset}/attack', exist_ok=True)

    print(f"\n{'='*80}")
    print(f"{'RUNG_percentile_adv_v2: Fixed Adversarial Training':^80}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading dataset...")
    A, X, y = get_dataset(args.dataset)
    sp = get_splits(y)
    train_idx, val_idx, test_idx = sp[0]

    print(f"  Dataset: {args.dataset}")
    print(f"  Nodes: {A.shape[0]}, Edges: {A.count_nonzero().item()//2}")
    print(f"  Classes: {y.max().item() + 1}\n")

    # Create model
    print("Creating RUNG_percentile_gamma model...")
    model_params = {
        'percentile_q': args.percentile_q,
        'use_layerwise_q': args.use_layerwise_q,
        'percentile_q_late': args.percentile_q_late,
    }
    model, _ = get_model_default(
        args.dataset,
        'RUNG_percentile_gamma',
        custom_model_params=model_params,
    )
    print(f"  Model: {model.__class__.__name__}\n")

    # Train
    print("PHASE 1: ADVERSARIAL TRAINING\n")
    train_results = train_percentile_adv_v2(
        model,
        A, X, y,
        train_idx, val_idx, test_idx,
        max_epoch=args.max_epoch,
        alpha=args.alpha,
        train_pgd_steps=args.train_pgd_steps,
        warmup_epochs=args.warmup_epochs,
        attack_freq=args.attack_freq,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )

    results = {
        'dataset': args.dataset,
        'model': 'RUNG_percentile_adv_v2',
        'train': {
            'best_val_acc': train_results['best_val_acc'],
            'test_acc_clean': train_results['test_acc_clean'],
        },
        'attack': {}
    }

    # Attack (optional)
    if not args.skip_attack:
        print(f"\n{'='*80}")
        print(f"{'PHASE 2: PGD ATTACKS':^80}")
        print(f"{'='*80}\n")
        print(f"Running PGD attacks with budgets: {ATTACK_BUDGETS}")
        print(f"Attack steps: {args.attack_steps}\n")

        for budget in ATTACK_BUDGETS:
            print(f"  Budget {budget:.2f}... ", end='', flush=True)
            try:
                acc = attack_pgd_v2(
                    model,
                    A, X, y, test_idx,
                    budget=budget,
                    n_steps=args.attack_steps,
                    device=args.device,
                )
                results['attack'][budget] = acc
                print(f"accuracy={acc:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                results['attack'][budget] = None

    # Print summary
    print(f"\n{'='*80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'='*80}\n")

    print(f"Dataset: {args.dataset}")
    print(f"\nTraining:")
    print(f"  Best val accuracy:   {results['train']['best_val_acc']:.4f}")
    print(f"  Test accuracy (clean): {results['train']['test_acc_clean']:.4f}")

    if results['attack']:
        print(f"\nAttacks:")
        for budget in sorted(results['attack'].keys()):
            acc = results['attack'][budget]
            if acc is not None:
                clean_acc = results['train']['test_acc_clean']
                delta = clean_acc - acc
                print(f"  Budget {budget:.2f}: accuracy={acc:.4f} (Δ={delta:+.4f})")
            else:
                print(f"  Budget {budget:.2f}: ERROR")

    print(f"\n{'='*80}\n")

    return results


if __name__ == '__main__':
    main()
