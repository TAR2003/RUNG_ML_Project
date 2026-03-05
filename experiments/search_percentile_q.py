#!/usr/bin/env python
"""
experiments/search_percentile_q.py

Grid search over percentile_q values to find the best setting for
RUNG_percentile_gamma.

This is the primary tuning experiment.  Run this first to select
the best q before any full comparison experiment.

The search covers:
    q in {0.50, 0.60, 0.75, 0.85, 0.90, 0.95}

For each q, measures:
    - Clean accuracy (no attack) — 5-split average
    - Attacked accuracy at budgets: 0.05, 0.10, 0.20, 0.30, 0.40, 0.60
    - Std across splits (key metric: should be much lower than learnable gamma)

Results are saved progressively to CSV so partial results survive crashes.

Usage:
    python experiments/search_percentile_q.py --dataset cora
    python experiments/search_percentile_q.py --dataset citeseer
    python experiments/search_percentile_q.py --dataset cora --q_values 0.75 0.85 0.90
"""

import sys
import os
import copy
import csv
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from train_eval_data.get_dataset import get_dataset, get_splits
from train_eval_data.fit_percentile_gamma import fit_percentile_gamma
from model.rung_percentile_gamma import RUNG_percentile_gamma
from utils import accuracy

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_Q_VALUES = [0.50, 0.60, 0.75, 0.85, 0.90, 0.95]
ATTACK_BUDGETS   = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
NUM_SPLITS       = 5     # use first 5 splits from get_splits()


# ---------------------------------------------------------------------------
# Attack helper
# ---------------------------------------------------------------------------

def _run_attack(model, A_clean, X, y, test_idx, budget_fraction, device):
    """
    Run PGD evasion attack and return attacked accuracy.

    Imports attack utilities from gb (same as attack.py) only when needed.
    Returns clean accuracy if budget == 0.
    """
    if budget_fraction == 0.0:
        model.eval()
        with torch.no_grad():
            acc = accuracy(model(A_clean, X)[test_idx], y[test_idx]).item()
        return acc

    from gb.attack.gd import proj_grad_descent
    from gb.metric import margin
    from gb.pert import edge_diff_matrix

    model.eval()
    num_edges  = int(A_clean.sum().item() / 2)
    budget_int = max(1, int(budget_fraction * num_edges))

    # Build surrogate loss for PGD (same pattern as attack.py)
    def loss_fn(flip):
        A_pert = A_clean + edge_diff_matrix(flip, A_clean)
        return -margin(model(A_pert, X)[test_idx], y[test_idx]).sum()

    with torch.enable_grad():
        flip = proj_grad_descent(
            loss_fn, A_clean, budget_int,
            steps=100, step_size=0.1, momentum=0.0,
        )

    A_pert = A_clean + edge_diff_matrix(flip.detach(), A_clean)
    with torch.no_grad():
        acc = accuracy(model(A_pert, X)[test_idx], y[test_idx]).item()
    return acc


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def run_q_search(
    dataset_name: str,
    q_values=None,
    budgets=None,
    num_splits: int = NUM_SPLITS,
    hidden_dims=None,
    lam_hat: float = 0.9,
    scad_a: float = 3.7,
    prop_step: int = 10,
    lr: float = 5e-2,
    weight_decay: float = 5e-4,
    max_epoch: int = 300,
    results_dir: str = None,
    run_attack: bool = True,
) -> str:
    """
    Grid search over q values for RUNG_percentile_gamma.

    Saves results to CSV progressively — partial results are preserved
    even if the run is interrupted.

    Args:
        dataset_name:  'cora', 'citeseer', etc.
        q_values:      List of percentile_q values to search.
        budgets:       List of attack budget fractions (0.0 = clean).
        num_splits:    Number of data splits to average over.
        hidden_dims:   MLP hidden layer widths (default [64]).
        lam_hat:       Skip-connection fraction.
        scad_a:        SCAD shape parameter.
        prop_step:     Number of propagation layers (default 10).
        lr:            Learning rate.
        weight_decay:  L2 regularisation.
        max_epoch:     Max training epochs per trial.
        results_dir:   Directory for output CSV (default: results/comparison).
        run_attack:    If False, evaluate clean accuracy only (faster).

    Returns:
        csv_path: Path to results CSV file.
    """
    if q_values is None:
        q_values = DEFAULT_Q_VALUES
    if budgets is None:
        budgets = ATTACK_BUDGETS if run_attack else [0.0]
    if hidden_dims is None:
        hidden_dims = [64]
    if results_dir is None:
        results_dir = str(ROOT / "results" / "comparison")

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path  = os.path.join(
        results_dir,
        f'q_search_{dataset_name}_{timestamp}.csv'
    )

    fieldnames = ['dataset', 'q', 'split', 'budget', 'clean_acc', 'attacked_acc']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 60}")
    print(f"  Q Search: RUNG_percentile_gamma on {dataset_name}")
    print(f"  Q values: {q_values}")
    print(f"  Budgets:  {budgets}")
    print(f"  Splits:   {num_splits}")
    print(f"  Device:   {device}")
    print(f"  Results → {csv_path}")
    print(f"{'=' * 60}\n")

    # Load dataset once
    A_all, X_all, y_all = get_dataset(dataset_name)
    splits = get_splits(y_all)
    splits = splits[:num_splits]

    D = X_all.shape[1]
    C = int(y_all.max().item()) + 1

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for q in q_values:
            print(f"\n{'─' * 40}")
            print(f"  q = {q:.2f}")
            print(f"{'─' * 40}")

            split_clean_accs = []

            for split_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
                print(f"  Split {split_idx + 1}/{num_splits}  (q={q:.2f}) ...", end=' ')

                # Move tensors to device
                A   = A_all.to(device)
                X   = X_all.to(device)
                y   = y_all.to(device)
                tr  = train_idx.to(device)
                vl  = val_idx.to(device)
                te  = test_idx.to(device)

                # Build and train model
                torch.manual_seed(split_idx)
                model = RUNG_percentile_gamma(
                    in_dim       = D,
                    out_dim      = C,
                    hidden_dims  = hidden_dims,
                    lam_hat      = lam_hat,
                    percentile_q = q,
                    scad_a       = scad_a,
                    prop_step    = prop_step,
                ).to(device)

                fit_percentile_gamma(
                    model, A, X, y, tr, vl,
                    lr=lr, weight_decay=weight_decay,
                    max_epoch=max_epoch,
                    log_gamma_every=0,    # suppress per-epoch gamma logs
                )

                model.eval()

                # Clean accuracy
                with torch.no_grad():
                    clean_acc = accuracy(model(A, X)[te], y[te]).item()
                split_clean_accs.append(clean_acc)

                print(f"clean={clean_acc:.4f}")

                # Evaluate at each budget
                for budget in budgets:
                    if budget == 0.0:
                        atk_acc = clean_acc
                    else:
                        if run_attack:
                            atk_acc = _run_attack(
                                model, A, X, y, te, budget, device
                            )
                        else:
                            atk_acc = float('nan')

                    writer.writerow({
                        'dataset':      dataset_name,
                        'q':            q,
                        'split':        split_idx,
                        'budget':       budget,
                        'clean_acc':    clean_acc,
                        'attacked_acc': atk_acc,
                    })
                    f.flush()

            mean_clean = np.mean(split_clean_accs)
            std_clean  = np.std(split_clean_accs)
            print(f"  q={q:.2f}  mean_clean={mean_clean:.4f} ± {std_clean:.4f}")

    print(f"\n{'=' * 60}")
    print(f"  Q search complete → {csv_path}")
    print(f"{'=' * 60}\n")
    return csv_path


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def summarise_q_search(csv_path: str, budget: float = 0.40) -> None:
    """
    Print a summary table of mean ± std accuracy at a specific attack budget.

    Args:
        csv_path: Path to CSV output by run_q_search().
        budget:   Budget fraction to summarise (default 0.40).
    """
    import csv as csv_mod
    rows = []
    with open(csv_path, 'r') as f:
        for row in csv_mod.DictReader(f):
            rows.append(row)

    from collections import defaultdict
    by_q = defaultdict(list)
    for row in rows:
        b = float(row['budget'])
        if abs(b - budget) < 1e-6:
            by_q[float(row['q'])].append(float(row['attacked_acc']))

    print(f"\nQ Search Summary — budget={budget:.0%}")
    print(f"{'q':>6}  {'mean':>8}  {'std':>8}  {'n':>4}")
    print(f"{'─'*32}")
    for q in sorted(by_q.keys()):
        vals = by_q[q]
        print(
            f"  {q:.2f}  {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}  "
            f"{len(vals):>4}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Grid search over percentile_q for RUNG_percentile_gamma.'
    )
    parser.add_argument('--dataset',    type=str,   default='cora',
                        help='Dataset name (default: cora).')
    parser.add_argument('--q_values',   type=float, nargs='+',
                        default=DEFAULT_Q_VALUES,
                        help='Q values to search (default: 0.50 0.60 0.75 0.85 0.90 0.95).')
    parser.add_argument('--num_splits', type=int,   default=NUM_SPLITS,
                        help='Number of data splits (default: 5).')
    parser.add_argument('--max_epoch',  type=int,   default=300,
                        help='Max training epochs per trial (default: 300).')
    parser.add_argument('--lr',         type=float, default=5e-2,
                        help='Learning rate (default: 5e-2).')
    parser.add_argument('--no_attack',  action='store_true',
                        help='Skip attack evaluation (clean accuracy only, faster).')
    parser.add_argument('--results_dir', type=str,  default=None,
                        help='Output directory for results CSV.')
    parser.add_argument('--summary_budget', type=float, default=0.40,
                        help='Budget fraction for summary table (default: 0.40).')
    args = parser.parse_args()

    csv_path = run_q_search(
        dataset_name = args.dataset,
        q_values     = args.q_values,
        num_splits   = args.num_splits,
        max_epoch    = args.max_epoch,
        lr           = args.lr,
        results_dir  = args.results_dir,
        run_attack   = not args.no_attack,
    )

    summarise_q_search(csv_path, budget=args.summary_budget)
