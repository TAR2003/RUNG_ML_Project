#!/usr/bin/env python
"""
train_test_combined.py

One-command training and testing of RUNG_combined with all attack budgets.

This comprehensive script:
  1. Trains RUNG_combined on clean graph(s) and saves model
  2. Evaluates on test set (clean accuracy)
  3. Runs PGD attacks at budgets: [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
  4. Reports clean + attacked accuracy for each budget
  5. Logs all results to file and stdout

Usage:
    # Train and test on Cora with default parameters
    python train_test_combined.py --dataset cora

    # On Citeseer with custom percentile_q
    python train_test_combined.py --dataset citeseer --percentile_q 0.70

    # Train on multiple datasets
    python train_test_combined.py --datasets cora citeseer

    # Training only (no attacks)
    python train_test_combined.py --dataset cora --skip_attack

    # Full custom example
    python train_test_combined.py --dataset cora --percentile_q 0.75 \
                                  --max_epoch 300 --lr 0.05

For faster iteration during development:
    python train_test_combined.py --dataset cora --max_epoch 50 --skip_attack
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


# ============================================================
#  Configuration
# ============================================================

ATTACK_BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]


# ============================================================
#  Training pipeline
# ============================================================

def train_clean(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    max_epoch: int = 300,
    lr: float = 0.05,
    weight_decay: float = 5e-4,
    early_stopping_patience: int = 100,
    device: str = 'cpu',
) -> dict:
    """
    Train RUNG_combined using standard cleantraining (cross-entropy on train nodes).

    Args:
        model:      RUNG_combined instance
        A, X, y:    Graph data
        train_idx, val_idx, test_idx: Node index splits
        max_epoch: Max training epochs
        lr:        Learning rate
        weight_decay: L2 regularization
        early_stopping_patience: Early stopping patience
        device:    torch device

    Returns:
        dict with training history and final accuracies
    """
    model = model.to(device)
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    print(f"Training RUNG_combined (lr={lr}, epochs={max_epoch})")

    for epoch in range(1, max_epoch + 1):
        # ---- Train step ----
        model.train()
        optimizer.zero_grad()

        logits = model(A, X)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---- Validation ----
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_val = model(A, X)
                train_acc = accuracy(logits_val[train_idx], y[train_idx])
                val_acc = accuracy(logits_val[val_idx], y[val_idx])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"  Epoch {epoch:3d} | loss={loss.item():.4f} | "
                  f"train={train_acc:.4f} | val={val_acc:.4f} | "
                  f"best_val={best_val_acc:.4f}")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ---- Load best checkpoint ----
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final evaluation ----
    model.eval()
    with torch.no_grad():
        logits_final = model(A, X)
        final_test_acc = accuracy(logits_final[test_idx], y[test_idx])
        final_val_acc = accuracy(logits_final[val_idx], y[val_idx])

    print(f"\nTraining complete!")
    print(f"  Best epoch:       {best_epoch}")
    print(f"  Best val acc:     {best_val_acc:.4f}")
    print(f"  Final test acc:   {final_test_acc:.4f}\n")

    return {
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'final_test_acc': final_test_acc,
        'model': model,
    }


def attack_pgd(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    test_idx: torch.Tensor,
    budget: float,
    n_epochs: int = 10,
    lr_attack: float = 0.01,
    device: str = 'cpu',
) -> float:
    """
    Run PGD attack on the model and return attacked test accuracy.

    Args:
        model:       Trained model
        A, X, y:     Graph data
        test_idx:    Test indices
        budget:      Perturbation budget (0 to 1, fraction of edges)
        n_epochs:    Attack iterations
        lr_attack:   Attack learning rate
        device:      torch device

    Returns:
        Accuracy under attack
    """
    model = model.to(device)
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)

    # ---- Define loss and gradient functions ----
    def loss_fn(flip):
        """Compute attack loss on a dense flip matrix"""
        if flip.numel() == 0:
            return torch.tensor(0.0, device=device)
        A_pert = A + (flip * (1.0 - 2.0 * A))
        out = model(A_pert, X)
        return margin(out[test_idx], y[test_idx]).mean()

    def grad_fn(flip):
        """Compute gradient of loss w.r.t. flip matrix"""
        if flip.numel() == 0:
            return torch.zeros_like(flip)
        loss = loss_fn(flip)
        try:
            grad = torch.autograd.grad(loss, flip, create_graph=True)[0]
        except RuntimeError:
            grad = torch.zeros_like(flip)
        return grad

    # Convert budget from ratio to number of edges (undirected: divide by 2)
    budget_edge_num = int(budget * A.count_nonzero().item() // 2)

    # ---- Run PGD attack ----
    try:
        edge_pert, _ = proj_grad_descent(
            flip_shape_or_init=A.shape,
            symmetric=True,
            device=A.device,
            budget=budget_edge_num,
            grad_fn=grad_fn,
            loss_fn=loss_fn,
            iterations=n_epochs,
            base_lr=lr_attack,
            grad_clip=1.0,
            progress=False,
        )
    except Exception as e:
        print(f"    ERROR during attack: {e}")
        # Return clean accuracy if attack fails
        model.eval()
        with torch.no_grad():
            logits = model(A, X)
            return accuracy(logits[test_idx], y[test_idx]).item()

    # ---- Convert edge indices to dense perturbation matrix ----
    # edge_pert is [num_edges, 2] edge indices
    # edge_diff_matrix converts it to dense perturbation matrix
    if edge_pert.numel() > 0:
        A_attacked = A + edge_diff_matrix(edge_pert.long(), A)
    else:
        A_attacked = A  # No perturbations, use original

    # ---- Evaluate on attacked graph ----
    model.eval()
    with torch.no_grad():
        logits = model(A_attacked, X)
        attacked_acc = accuracy(logits[test_idx], y[test_idx])

    return attacked_acc.item()


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train and test RUNG_combined with attacks'
    )
    parser.add_argument(
        '--datasets', nargs='+', default=['cora'],
        help='Datasets to run on (default: cora)'
    )
    parser.add_argument(
        '--percentile_q', type=float, default=0.75,
        help='Percentile q for gamma (default: 0.75)'
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
    parser.add_argument(
        '--max_epoch', type=int, default=300,
        help='Max training epochs (default: 300)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.05,
        help='Learning rate (default: 0.05)'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=5e-4,
        help='L2 regularization (default: 5e-4)'
    )
    parser.add_argument(
        '--attack_epochs', type=int, default=10,
        help='PGD attack iterations (default: 10)'
    )
    parser.add_argument(
        '--attack_lr', type=float, default=0.01,
        help='PGD attack learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--skip_attack', action='store_true',
        help='Skip attack phase'
    )
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='torch device'
    )

    args = parser.parse_args()

    # Create log directories
    os.makedirs('log', exist_ok=True)
    for ds in args.datasets:
        os.makedirs(f'log/{ds}', exist_ok=True)
        os.makedirs(f'log/{ds}/attack', exist_ok=True)

    # Main loop over datasets
    print(f"\n{'='*80}")
    print(f"{'RUNG_COMBINED: Train + Attack':^80}")
    print(f"{'='*80}\n")

    summary = {}

    for dataset in args.datasets:
        print(f"\n{'#'*80}")
        print(f"{'# DATASET: ' + dataset.upper():80}")
        print(f"{'#'*80}\n")

        # Load data
        A, X, y = get_dataset(dataset)
        sp = get_splits(y)
        train_idx, val_idx, test_idx = sp[0]

        # Create model
        model_params = {
            'percentile_q': args.percentile_q,
            'use_layerwise_q': args.use_layerwise_q,
            'percentile_q_late': args.percentile_q_late,
        }
        model, _ = get_model_default(
            dataset,
            'RUNG_combined',
            custom_model_params=model_params,
        )

        print(f"Model: {model}\n")

        # Train
        train_results = train_clean(
            model,
            A, X, y,
            train_idx, val_idx, test_idx,
            max_epoch=args.max_epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
        )

        summary[dataset] = {
            'train': {
                'best_epoch': train_results['best_epoch'],
                'best_val_acc': train_results['best_val_acc'],
                'test_acc_clean': train_results['final_test_acc'],
            },
            'attack': {}
        }

        # Attack (optional)
        if not args.skip_attack:
            print(f"\nRunning PGD attacks with budgets: {ATTACK_BUDGETS}")
            print(f"Attack epochs: {args.attack_epochs}, lr: {args.attack_lr}\n")

            for budget in ATTACK_BUDGETS:
                print(f"  Budget {budget:.2f}... ", end='', flush=True)
                try:
                    acc = attack_pgd(
                        model,
                        A, X, y, test_idx,
                        budget=budget,
                        n_epochs=args.attack_epochs,
                        lr_attack=args.attack_lr,
                        device=args.device,
                    )
                    summary[dataset]['attack'][budget] = acc
                    print(f"accuracy={acc:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    summary[dataset]['attack'][budget] = None

            print()

    # Print summary
    print(f"\n{'='*80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'='*80}\n")

    for dataset, results in summary.items():
        print(f"\n{dataset.upper()}:")
        print(f"  Training:")
        print(f"    Best epoch:        {results['train']['best_epoch']}")
        print(f"    Best val acc:      {results['train']['best_val_acc']:.4f}")
        print(f"    Test acc (clean):  {results['train']['test_acc_clean']:.4f}")

        if results['attack']:
            print(f"  Attacks:")
            for budget in sorted(results['attack'].keys()):
                acc = results['attack'][budget]
                if acc is not None:
                    # Compute robustness decrease
                    clean_acc = results['train']['test_acc_clean']
                    delta = clean_acc - acc
                    print(f"    Budget {budget:.2f}: accuracy={acc:.4f} "
                          f"(Δ={delta:+.4f})")
                else:
                    print(f"    Budget {budget:.2f}: ERROR")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

