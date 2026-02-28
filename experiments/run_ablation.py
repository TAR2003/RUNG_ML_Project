"""
Systematic ablation experiment runner for RUNG extensions.

Runs experiments across:
- Multiple penalty functions (mcp, scad, l1, l2, adaptive)
- Multiple datasets (homophilic + heterophilic)
- Multiple attack budgets
- Multiple random seeds

Saves results to CSV incrementally (safe against interruptions).

Usage:
    python experiments/run_ablation.py --experiment penalty_comparison
    python experiments/run_ablation.py --experiment heterophilic
    python experiments/run_ablation.py --experiment bias_curve
    python experiments/run_ablation.py --experiment gamma_sensitivity
    python experiments/run_ablation.py --experiment num_layers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import copy
import csv
import itertools
import time
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch

# Core codebase imports
from train_eval_data.get_dataset import get_dataset, get_splits
from train_eval_data.fit import fit
from exp.config.get_model import get_model_default
from utils import accuracy as _accuracy
from gb.attack.gd import proj_grad_descent
from gb.metric import margin
from gb.pert import edge_diff_matrix

# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENTS: Dict[str, Dict[str, Any]] = {

    'penalty_comparison': {
        'description': 'Compare MCP vs SCAD vs l1 vs l2 on homophilic datasets',
        'datasets':    ['cora', 'citeseer'],
        'penalties':   ['L2', 'L1', 'MCP', 'SCAD'],
        'gammas':      [6.0],           # Fixed gamma for fair comparison
        'lambda_hats': [0.9],           # Fixed lam_hat
        'budgets':     [0, 5, 10, 20, 30, 40],   # Global attack budgets (% of edges)
        'attack_type': 'global',
        'seeds':       [0, 1, 2, 3, 4],
        'num_layers':  10,
        'measure_bias': False,
    },

    'heterophilic': {
        'description': 'Test all penalties on heterophilic datasets',
        'datasets':    ['chameleon', 'squirrel', 'actor', 'cornell', 'texas'],
        'penalties':   ['L1', 'MCP', 'ADAPTIVE'],
        'gammas':      [0.5, 1.0, 2.0, 3.0],
        'lambda_hats': [0.9],
        'budgets':     [0, 5, 10, 20],
        'attack_type': 'global',
        'seeds':       [0, 1, 2, 3, 4],
        'num_layers':  10,
        'measure_bias': False,
    },

    'bias_curve': {
        'description': 'Replicate and extend Figure 6: estimation bias vs attack budget',
        'datasets':    ['cora', 'chameleon'],    # One homo + one hetero
        'penalties':   ['L1', 'MCP', 'SCAD'],
        'gammas':      [6.0],
        'lambda_hats': [0.9],
        'budgets':     [0, 5, 10, 20, 30, 40],  # Global attack budgets (% of edges)
        'attack_type': 'global',
        'seeds':       [0, 1, 2],
        'num_layers':  10,
        'measure_bias': True,
    },

    'gamma_sensitivity': {
        'description': 'Replicate and extend Figure 12: gamma vs lambda heatmap',
        'datasets':    ['cora'],
        'penalties':   ['MCP'],
        'gammas':      [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
        'lambda_hats': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        'budgets':     [0, 10, 20, 40],
        'attack_type': 'global',
        'seeds':       [0, 1, 2],
        'num_layers':  10,
        'measure_bias': False,
    },

    'num_layers': {
        'description': 'Replicate Figure 13: performance vs number of propagation layers',
        'datasets':    ['cora'],
        'penalties':   ['MCP', 'L1'],
        'gammas':      [6.0],
        'lambda_hats': [0.9],
        'budgets':     [0, 10, 20, 40],
        'attack_type': 'global',
        'seeds':       [0, 1, 2],
        'num_layers':  [1, 2, 3, 5, 7, 10, 15, 20],  # Vary layers
        'measure_bias': False,
    },
}


# ---------------------------------------------------------------------------
# Attack utilities
# ---------------------------------------------------------------------------

def make_A_pert(A: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
    """Apply edge flip to adjacency matrix."""
    return A + edge_diff_matrix(flip, A)


def pgd_attack(
    model: torch.nn.Module,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    test_idx: torch.Tensor,
    budget_edge_num: int,
    iterations: int = 200,
) -> torch.Tensor:
    """
    Run PGD global evasion attack.

    Args:
        model:            Trained model (in eval mode)
        A:                [N, N] clean adjacency
        X:                [N, F] features
        y:                [N] labels
        test_idx:         Test node indices
        budget_edge_num:  Max number of edge perturbations
        iterations:       PGD iterations

    Returns:
        A_pert: [N, N] perturbed adjacency
    """
    model.eval()

    def loss_fn(flip):
        A_pert = A + flip * (1 - 2 * A)
        out = model(A_pert, X)
        return margin(out[test_idx, :], y[test_idx]).tanh().mean()

    def grad_fn(flip):
        loss = loss_fn(flip)
        return torch.autograd.grad(loss, flip)[0]

    flip, _loss = proj_grad_descent(
        A.shape, True, A.device, budget_edge_num, grad_fn, loss_fn,
        grad_clip=1, iterations=iterations
    )
    return make_A_pert(A, flip)


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a RUNG model, optionally attack it, and return accuracy + bias.

    Args:
        config: Dict with keys:
            dataset      — dataset name (str)
            penalty      — norm/penalty name, e.g. 'MCP', 'SCAD', 'L1', 'L2', 'ADAPTIVE'
            gamma        — float, penalty threshold
            lambda_hat   — float, skip-connection strength (not currently exposed
                           in get_model_default, uses fixed 0.9 — extend if needed)
            budget       — int, attack budget as % of total edges (0 = clean)
            attack_type  — 'global' (only global PGD supported here)
            seed         — int, random seed
            num_layers   — int, number of propagation layers
            measure_bias — bool, whether to compute estimation bias

    Returns:
        Dict with keys: accuracy, bias_total, bias_mean
    """
    dataset    = config['dataset']
    penalty    = config['penalty']
    gamma      = config['gamma']
    budget_pct = config['budget']
    seed       = config['seed']
    num_layers = config['num_layers']
    measure_bias = config.get('measure_bias', False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Load data ---------------------------------------------------------
    A, X, y = get_dataset(dataset)
    sps = get_splits(y)

    # ---- Build model -------------------------------------------------------
    model, fit_params = get_model_default(
        dataset,
        model_name='RUNG',
        custom_model_params={'gamma': gamma, 'norm': penalty},
        custom_fit_params={'lr': 5e-2, 'weight_decay': 5e-4, 'max_epoch': 300},
        seed=seed,
    )
    # Override propagation layers if needed
    model.prop_layer_num = num_layers
    model = model.to(device)

    # ---- Train and evaluate across splits ----------------------------------
    accuracies = []
    bias_totals = []

    for split_idx, (train_idx, val_idx, test_idx) in enumerate(sps):
        # Fresh copy per split
        cur_model = copy.deepcopy(model)
        torch.manual_seed(seed * 100 + split_idx)

        # Move data to device
        A_dev = A.to(device)
        X_dev = X.to(device)
        y_dev = y.to(device)

        # Train on clean graph
        fit(cur_model, A_dev, X_dev, y_dev, train_idx.to(device), val_idx.to(device),
            lr=5e-2, weight_decay=5e-4, max_epoch=300)

        # Generate attack if budget > 0
        if budget_pct > 0:
            total_edges = int(A.count_nonzero().item() // 2)
            budget_edges = max(1, int(budget_pct / 100.0 * total_edges))

            cur_model.eval()
            A_pert = pgd_attack(
                cur_model, A_dev, X_dev, y_dev, test_idx.to(device), budget_edges
            )
        else:
            A_pert = A_dev

        # Evaluate accuracy
        cur_model.eval()
        with torch.no_grad():
            logits = cur_model(A_pert, X_dev)
            acc = _accuracy(logits[test_idx.to(device)], y_dev[test_idx.to(device)])
        accuracies.append(acc.item())

        # Compute estimation bias if requested
        if measure_bias and budget_pct > 0:
            from utils.metrics import compute_estimation_bias
            bias_total, _bias_mean = compute_estimation_bias(
                cur_model, A_dev, X_dev, A_pert, device
            )
            bias_totals.append(bias_total)

    result = {
        'accuracy':   float(np.mean(accuracies)),
        'std':        float(np.std(accuracies)),
        'bias_total': float(np.mean(bias_totals)) if bias_totals else float('nan'),
        'bias_mean':  float('nan'),
    }
    return result


# ---------------------------------------------------------------------------
# Full ablation runner
# ---------------------------------------------------------------------------

def run_ablation(experiment_name: str, results_dir: str = './results') -> str:
    """
    Run a full ablation study for a named experiment configuration.

    Saves results to CSV progressively (work is not lost if the run crashes).

    Args:
        experiment_name: Key from the EXPERIMENTS dict
        results_dir:     Directory for output CSV files

    Returns:
        csv_path: Path to the saved CSV file
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: '{experiment_name}'. "
            f"Choose from: {list(EXPERIMENTS.keys())}"
        )

    config = EXPERIMENTS[experiment_name]
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(results_dir, f'{experiment_name}_{timestamp}.csv')

    print(f"\n{'='*60}")
    print(f"Experiment : {experiment_name}")
    print(f"Description: {config['description']}")
    print(f"Results CSV: {csv_path}")
    print(f"{'='*60}\n")

    # Build all (dataset, penalty, gamma, lambda_hat, budget, seed, num_layers) combos
    num_layers_list = (
        config['num_layers']
        if isinstance(config['num_layers'], list)
        else [config['num_layers']]
    )

    all_configs = list(itertools.product(
        config['datasets'],
        config['penalties'],
        config['gammas'],
        config['lambda_hats'],
        config['budgets'],
        config['seeds'],
        num_layers_list,
    ))
    print(f"Total configurations: {len(all_configs)}\n")

    fieldnames = [
        'dataset', 'penalty', 'gamma', 'lambda_hat', 'budget',
        'seed', 'num_layers', 'accuracy', 'std',
        'bias_total', 'bias_mean', 'runtime_seconds',
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, (dataset, penalty, gamma, lambda_hat, budget, seed, num_layers) \
                in enumerate(all_configs, start=1):

            print(f"[{i}/{len(all_configs)}] "
                  f"dataset={dataset} penalty={penalty} gamma={gamma} "
                  f"lam_hat={lambda_hat} budget={budget}% "
                  f"seed={seed} layers={num_layers}")

            start_time = time.time()
            try:
                run_cfg = {
                    'dataset':      dataset,
                    'penalty':      penalty,
                    'gamma':        gamma,
                    'lambda_hat':   lambda_hat,
                    'budget':       budget,
                    'attack_type':  config['attack_type'],
                    'seed':         seed,
                    'num_layers':   num_layers,
                    'measure_bias': config.get('measure_bias', False),
                }
                result = run_single_experiment(run_cfg)
                runtime = time.time() - start_time

                row = {
                    'dataset':         dataset,
                    'penalty':         penalty,
                    'gamma':           gamma,
                    'lambda_hat':      lambda_hat,
                    'budget':          budget,
                    'seed':            seed,
                    'num_layers':      num_layers,
                    'accuracy':        result['accuracy'],
                    'std':             result['std'],
                    'bias_total':      result['bias_total'],
                    'bias_mean':       result['bias_mean'],
                    'runtime_seconds': runtime,
                }
                writer.writerow(row)
                csvfile.flush()  # write incrementally

                print(f"  → accuracy={result['accuracy']:.4f} ± {result['std']:.4f}, "
                      f"bias={result['bias_total']:.4f}, "
                      f"time={runtime:.1f}s")

            except Exception as exc:
                runtime = time.time() - start_time
                print(f"  → FAILED: {exc}")
                writer.writerow({
                    'dataset': dataset, 'penalty': penalty, 'gamma': gamma,
                    'lambda_hat': lambda_hat, 'budget': budget, 'seed': seed,
                    'num_layers': num_layers, 'accuracy': float('nan'),
                    'std': float('nan'), 'bias_total': float('nan'),
                    'bias_mean': float('nan'), 'runtime_seconds': runtime,
                })
                csvfile.flush()

    print(f"\n{'='*60}")
    print(f"Complete. Results: {csv_path}")
    print(f"{'='*60}\n")
    return csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run RUNG ablation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(
            f"  {name:22s}— {cfg['description']}"
            for name, cfg in EXPERIMENTS.items()
        )
    )
    parser.add_argument(
        '--experiment', type=str, required=True,
        choices=list(EXPERIMENTS.keys()),
        help='Which experiment to run (see descriptions above)',
    )
    parser.add_argument(
        '--results_dir', type=str, default='./results',
        help='Directory to save results CSV (default: ./results)',
    )
    args = parser.parse_args()

    run_ablation(args.experiment, args.results_dir)
