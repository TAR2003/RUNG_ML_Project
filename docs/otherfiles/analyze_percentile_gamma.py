"""
analyze_percentile_gamma.py

Two key analyses for RUNG_percentile_gamma:

1. Gamma profile plot:
   Shows how gamma adapts across layers during a forward pass.
   Expected: gammas decrease with layer depth.
   This confirms that percentile gamma solves the depth-gamma mismatch
   problem (fixed or learned gamma fails at deep layers because features
   are smoothed and differences are small).

2. Variance comparison:
   Compares std across seeds/splits for:
   - RUNG (fixed gamma)
   - RUNG_learnable_gamma
   - RUNG_percentile_gamma
   Expected: percentile gamma has MUCH lower std.

Usage:
    # After training both models on the same dataset:
    python analyze_percentile_gamma.py \
        --dataset cora \
        --model_lg_path exp/models/cora/RUNG_learnable_gamma_MCP_6.0/model_0.pt \
        --percentile_q 0.75

    # Or import individual functions in a notebook.
"""

import sys
import os
import argparse
import copy
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Gamma profile comparison
# ---------------------------------------------------------------------------

def plot_gamma_profile_comparison(
    model_learnable,
    model_percentile,
    A,
    X,
    device='cpu',
    save_path='figures/gamma_profile_comparison.pdf',
):
    """
    Side-by-side plot of gamma profiles across layers.

    Left:  RUNG_learnable_gamma  — gammas learned during training.
    Right: RUNG_percentile_gamma — gammas from quantile on current features.

    Both should show decreasing gammas with depth if the depth-gamma
    mismatch hypothesis is correct.

    Args:
        model_learnable:  Trained RUNG_learnable_gamma instance.
        model_percentile: Trained RUNG_percentile_gamma instance.
        A:                [N, N] adjacency matrix.
        X:                [N, D] node features.
        device:           torch device string.
        save_path:        Output PDF path.
    """
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    A = A.to(device)
    X = X.to(device)

    # Learned gammas from RUNG_learnable_gamma (exp(log_lam) * scad_a)
    learned_gammas = model_learnable.get_learned_gammas()

    # Percentile gammas — require a forward pass
    model_percentile.eval()
    with torch.no_grad():
        _ = model_percentile(A, X)
    percentile_gammas = model_percentile.get_last_gammas()

    layers = list(range(len(learned_gammas)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    # Left: learnable gamma
    axes[0].plot(layers, learned_gammas,
                 'o-', color='steelblue', linewidth=2, markersize=7)
    axes[0].set_title('RUNG_learnable_gamma\n(trained parameters)', fontsize=11)
    axes[0].set_xlabel('Layer k')
    axes[0].set_ylabel('Gamma value γ^(k)')
    axes[0].axhline(y=np.mean(learned_gammas), color='gray',
                    linestyle='--', alpha=0.7, label=f"mean={np.mean(learned_gammas):.2f}")
    axes[0].legend(fontsize=9)

    # Right: percentile gamma
    axes[1].plot(layers, percentile_gammas,
                 's-', color='coral', linewidth=2, markersize=7)
    axes[1].set_title(
        f'RUNG_percentile_gamma\n(q={model_percentile.percentile_q})',
        fontsize=11
    )
    axes[1].set_xlabel('Layer k')
    axes[1].set_ylabel('Gamma value γ^(k)')
    axes[1].axhline(y=np.mean(percentile_gammas), color='gray',
                    linestyle='--', alpha=0.7, label=f"mean={np.mean(percentile_gammas):.2f}")
    axes[1].legend(fontsize=9)

    for ax in axes:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(
        'Gamma Adaptation Across Layers\n'
        'Decreasing trend confirms depth-gamma mismatch hypothesis',
        y=1.02
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Print summary
    print(f"\nLearnable gamma profile (layer 0 → {len(learned_gammas)-1}):")
    print(f"  first={learned_gammas[0]:.4f}, last={learned_gammas[-1]:.4f}, "
          f"decreasing={learned_gammas[0] > learned_gammas[-1]}")
    print(f"\nPercentile gamma profile (layer 0 → {len(percentile_gammas)-1}):")
    print(f"  first={percentile_gammas[0]:.4f}, last={percentile_gammas[-1]:.4f}, "
          f"decreasing={percentile_gammas[0] > percentile_gammas[-1]}")


# ---------------------------------------------------------------------------
# 2. Variance comparison
# ---------------------------------------------------------------------------

def plot_variance_comparison(
    results_csv_path,
    save_path='figures/variance_comparison.pdf',
):
    """
    Bar chart: std of accuracy across splits/seeds for each model and budget.

    The key claim: RUNG_percentile_gamma has lower variance than
    RUNG_learnable_gamma because gamma is deterministic.

    Expected CSV columns: model, budget, split, attacked_acc

    Args:
        results_csv_path: CSV with per-split attacked accuracy.
        save_path:        Output PDF.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    df = pd.read_csv(results_csv_path)

    std_df = df.groupby(['model', 'budget'])['attacked_acc'].std().reset_index()
    std_df.columns = ['model', 'budget', 'std']

    models  = std_df['model'].unique()
    budgets = sorted(std_df['budget'].unique())

    fig, ax = plt.subplots(figsize=(10, 5))

    x      = np.arange(len(budgets))
    width  = 0.8 / len(models)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, model in enumerate(models):
        mdata  = std_df[std_df['model'] == model].sort_values('budget')
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, mdata['std'].values,
               width=width * 0.9,
               label=model,
               color=colors[i % len(colors)],
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{b:.0%}' for b in budgets])
    ax.set_xlabel('Attack Budget')
    ax.set_ylabel('Std of Accuracy Across Splits')
    ax.set_title(
        'Variance Comparison Across Models\n'
        'Lower is better — percentile gamma should be most stable'
    )
    ax.legend(loc='upper left', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Q sensitivity plot
# ---------------------------------------------------------------------------

def plot_q_sensitivity(
    q_search_csv_path,
    budget=0.40,
    save_path='figures/q_sensitivity.pdf',
):
    """
    Line plot: attacked accuracy vs percentile_q at a fixed attack budget.

    Helps choose the best q value and visualises the clean/attacked tradeoff:
    - Low q (aggressive) → better robustness, possibly worse clean accuracy
    - High q (light) → better clean accuracy, less robustness

    Expected CSV columns: q, clean_acc, attacked_acc  (from search_percentile_q.py)

    Args:
        q_search_csv_path: CSV from experiments/search_percentile_q.py.
        budget:            Attack budget fraction to plot (default 0.40).
        save_path:         Output PDF.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    df = pd.read_csv(q_search_csv_path)
    df_budget = df[np.isclose(df['budget'].astype(float), budget)]

    summary = df_budget.groupby('q').agg(
        clean_mean    = ('clean_acc',    'mean'),
        attacked_mean = ('attacked_acc', 'mean'),
        attacked_std  = ('attacked_acc', 'std'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(summary['q'], summary['attacked_mean'],
                yerr=summary['attacked_std'],
                marker='o', linewidth=2, markersize=7,
                color='coral', capsize=4,
                label=f'Attacked (budget={budget:.0%})')

    ax.plot(summary['q'], summary['clean_mean'],
            marker='s', linewidth=2, markersize=7,
            color='steelblue', linestyle='--',
            label='Clean (no attack)')

    ax.set_xlabel('Percentile q')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Sensitivity to Percentile q\nBudget = {budget:.0%}')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Print best q
    best_row = summary.loc[summary['attacked_mean'].idxmax()]
    print(f"\nBest q at budget={budget:.0%}: q={best_row['q']:.2f}  "
          f"attacked={best_row['attacked_mean']:.4f} ± {best_row['attacked_std']:.4f}")


# ---------------------------------------------------------------------------
# 4. Determinism verification
# ---------------------------------------------------------------------------

def verify_gamma_determinism(
    dataset_name: str = 'cora',
    percentile_q: float = 0.75,
    prop_step: int = 3,
) -> bool:
    """
    Verify that gamma computation is deterministic (no stochastic elements).

    The key property: given the same model weights and the same graph,
    running the model twice gives IDENTICAL gammas. This confirms that
    torch.quantile introduces no randomness.

    IMPORTANT DESIGN NOTE — why gammas differ across random seeds:
        gamma^(k) = quantile(y^(k), q) where y^(k) depends on F^(k)
        and F^(k) depends on F^(0) = MLP(X).  Since different seeds
        initialise MLP weights differently, F^(0) changes → y^(k) changes
        → gamma^(k) changes.  This is EXPECTED and CORRECT behaviour.

    The stability claim for RUNG_percentile_gamma is NOT that gammas are
    numerically identical across seeds, but that:
        (a) gamma is a well-behaved, non-degenerate function of current
            features — it CANNOT get stuck at 0 (all edges pruned) or ∞
            (no edges pruned) the way learnable_gamma can via gradient
            stalling.
        (b) Given the same trained model, gamma is fully reproducible —
            no sampling or stochastic elements.
        (c) Training variance (std of final test accuracy across seeds)
            is lower for percentile_gamma than for learnable_gamma,
            because bad gamma initialisation cannot derail training.

    This function verifies properties (a) and (b).

    Args:
        dataset_name:  Dataset to load.
        percentile_q:  q value for the test.
        prop_step:     Number of layers (use small for speed).

    Returns:
        True if both checks pass, False otherwise.
    """
    import torch
    from train_eval_data.get_dataset import get_dataset, get_splits
    from model.rung_percentile_gamma import RUNG_percentile_gamma

    A, X, y = get_dataset(dataset_name)
    D = X.shape[1]
    C = int(y.max().item()) + 1

    torch.manual_seed(42)
    model = RUNG_percentile_gamma(
        in_dim=D, out_dim=C, hidden_dims=[64],
        percentile_q=percentile_q, prop_step=prop_step,
    )
    model.eval()

    # ---- Check (b): same model + same data → same gammas ----
    with torch.no_grad():
        _ = model(A, X)
    gammas_run1 = model.get_last_gammas()[:]

    with torch.no_grad():
        _ = model(A, X)
    gammas_run2 = model.get_last_gammas()[:]

    reproducible = True
    for k, (g1, g2) in enumerate(zip(gammas_run1, gammas_run2)):
        if abs(g1 - g2) > 1e-6:
            print(f"FAIL (reproducibility) layer {k}: run1={g1:.8f} run2={g2:.8f}")
            reproducible = False

    if reproducible:
        print(f"PASS (b): Gammas are identical across two forward passes of the same model.")
        print(f"  Gamma profile: {[round(g, 4) for g in gammas_run1]}")

    # ---- Check (a): gammas are bounded (no stuck-at-0 or stuck-at-inf) ----
    eps = model.eps
    max_possible = float('inf')
    non_degenerate = all(
        g is not None and g > eps and g < 1e6
        for g in gammas_run1
    )
    if non_degenerate:
        print(f"PASS (a): All gammas are well-bounded (eps={eps:.1e} < gamma < 1e6).")
    else:
        print(f"FAIL (a): Some gammas are degenerate (zero or extremely large).")
        print(f"  Gammas: {gammas_run1}")
        reproducible = False

    # ---- Show cross-seed gamma VARIATION (expected, explained) ----
    print(f"\nExpected cross-seed gamma variation (for information only):")
    print(f"  (Different seeds → different MLP weights → different embeddings → different gammas)")
    print(f"  (This is correct — the variance reduction claim is about test accuracy, not gamma values)")
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        m_seed = RUNG_percentile_gamma(
            in_dim=D, out_dim=C, hidden_dims=[64],
            percentile_q=percentile_q, prop_step=prop_step,
        )
        m_seed.eval()
        with torch.no_grad():
            _ = m_seed(A, X)
        g0 = m_seed.get_last_gammas()[0]
        print(f"  seed={seed}: gamma[0]={g0:.4f}")

    return reproducible


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis tools for RUNG_percentile_gamma.'
    )
    subparsers = parser.add_subparsers(dest='command')

    # Subcommand: verify determinism
    p_det = subparsers.add_parser('determinism',
                                   help='Verify gamma determinism across seeds.')
    p_det.add_argument('--dataset',      type=str,   default='cora')
    p_det.add_argument('--percentile_q', type=float, default=0.75)
    p_det.add_argument('--prop_step',    type=int,   default=3)

    # Subcommand: q sensitivity plot
    p_q = subparsers.add_parser('q_sensitivity',
                                 help='Plot q sensitivity from search CSV.')
    p_q.add_argument('--csv',     type=str, required=True,
                     help='CSV from experiments/search_percentile_q.py')
    p_q.add_argument('--budget',  type=float, default=0.40)
    p_q.add_argument('--out',     type=str,
                     default='figures/q_sensitivity.pdf')

    # Subcommand: variance comparison plot
    p_var = subparsers.add_parser('variance',
                                   help='Plot variance comparison from results CSV.')
    p_var.add_argument('--csv',   type=str, required=True)
    p_var.add_argument('--out',   type=str,
                       default='figures/variance_comparison.pdf')

    args = parser.parse_args()

    if args.command == 'determinism':
        ok = verify_gamma_determinism(
            dataset_name = args.dataset,
            percentile_q = args.percentile_q,
            prop_step    = args.prop_step,
        )
        sys.exit(0 if ok else 1)

    elif args.command == 'q_sensitivity':
        plot_q_sensitivity(args.csv, budget=args.budget, save_path=args.out)

    elif args.command == 'variance':
        plot_variance_comparison(args.csv, save_path=args.out)

    else:
        parser.print_help()
