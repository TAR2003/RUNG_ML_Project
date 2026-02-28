"""
Visualization module for RUNG experiment results.

Reproduces and extends key figures from the RUNG paper (NeurIPS 2024):
    Figure 2  → plot_bias_simulation()
    Figure 6  → plot_bias_curves()
    Figure 7  → plot_edge_diff_distribution()
    Figure 12 → plot_gamma_heatmap()
    Figure 13 → plot_layer_sensitivity()

New figures for the extensions in this repository:
    NEW-1     → plot_robustness_curves()       [penalty comparison]
    NEW-2     → plot_homophily_vs_performance() [hetero analysis]
    NEW-3     → plot_penalty_comparison_bars()  [clean accuracy summary]

Usage:
    python experiments/plot_results.py --figure robustness \\
        --results_csv results/penalty_comparison_20260227.csv

    python experiments/plot_results.py --figure bias_curves \\
        --results_csv results/bias_curve_20260227.csv

    python experiments/plot_results.py --figure homophily \\
        --results_csv results/heterophilic_20260227.csv

    python experiments/plot_results.py --figure bias_simulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Global plot style (NeurIPS-style aesthetics)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'legend.fontsize':   10,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'figure.dpi':        150,
    'axes.grid':         False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'lines.linewidth':   1.8,
    'lines.markersize':  6,
})

# Color palette
COLORS = {
    'L2':       '#d62728',   # Red    — l2 / GCN / APPNP
    'L1':       '#1f77b4',   # Blue   — l1 / biased baseline
    'MCP':      '#2ca02c',   # Green  — MCP / RUNG (paper default)
    'SCAD':     '#ff7f0e',   # Orange — SCAD
    'ADAPTIVE': '#9467bd',   # Purple — Adaptive
    'MLP':      '#8c564b',   # Brown  — MLP (no propagation)
}

MARKERS = {
    'L2':       's',
    'L1':       'o',
    'MCP':      '^',
    'SCAD':     'D',
    'ADAPTIVE': 'P',
}

LABELS = {
    'L2':       'l2 (APPNP)',
    'L1':       'l1 (RUNG-l1)',
    'MCP':      'MCP (RUNG)',
    'SCAD':     'SCAD',
    'ADAPTIVE': 'Adaptive',
}

# Approximate edge homophily ratios for plotting
HOMOPHILY_MAP = {
    'cora':      0.81,
    'citeseer':  0.74,
    'chameleon': 0.23,
    'squirrel':  0.22,
    'actor':     0.22,
    'cornell':   0.20,
    'texas':     0.11,
    'wisconsin': 0.21,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 2 reproduction: mean estimator bias simulation
# ---------------------------------------------------------------------------

def _weiszfeld(pts: np.ndarray, max_iter: int = 1000, eps: float = 1e-8) -> np.ndarray:
    """Weiszfeld algorithm for the geometric median (l1 mean estimator)."""
    x = pts.mean(axis=0)
    for _ in range(max_iter):
        dists = np.linalg.norm(pts - x, axis=1).clip(min=eps)
        weights = 1.0 / dists
        x_new = (pts * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(x_new - x) < 1e-7:
            break
        x = x_new
    return x


def _mcp_mean(pts: np.ndarray, gamma: float = 2.0, max_iter: int = 1000, eps: float = 1e-8) -> np.ndarray:
    """
    IRLS estimator with MCP penalty for the geometric mean.
    Solves  argmin_mu Σ_i ρ_γ(||pts_i - mu||)  via IRLS.
    """
    x = pts.mean(axis=0)
    for _ in range(max_iter):
        dists = np.linalg.norm(pts - x, axis=1).clip(min=eps)
        # MCP weight:  w_i = max(0, 1/(2*dist) - 1/(2*gamma))
        weights = np.maximum(0.0, 1.0 / (2.0 * dists) - 1.0 / (2.0 * gamma))
        w_sum = weights.sum()
        if w_sum < eps:
            break
        x_new = (pts * weights[:, None]).sum(axis=0) / w_sum
        if np.linalg.norm(x_new - x) < 1e-7:
            break
        x = x_new
    return x


def plot_bias_simulation(save_path: str = 'figures/fig_bias_simulation.pdf') -> None:
    """
    Reproduce Figure 2 from the RUNG paper: mean estimator bias with outliers.

    Shows l2, l1 (Weiszfeld), and MCP estimators for 15%, 30%, 45% outlier ratios.
    Clean samples:   N((0,0), I)
    Outlier samples: N((8,8), 0.5*I)
    """
    _ensure_dir(save_path)
    np.random.seed(42)

    outlier_ratios = [0.15, 0.30, 0.45]
    n_clean = 100
    true_mean = np.array([0.0, 0.0])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, ratio in zip(axes, outlier_ratios):
        n_outlier = max(1, int(n_clean * ratio / (1.0 - ratio)))
        clean   = np.random.multivariate_normal([0, 0], np.eye(2), n_clean)
        outlier = np.random.multivariate_normal([8, 8], 0.5 * np.eye(2), n_outlier)
        all_pts = np.vstack([clean, outlier])

        l2_mean  = all_pts.mean(axis=0)
        l1_mean  = _weiszfeld(all_pts)
        mcp_mean = _mcp_mean(all_pts, gamma=2.0)

        ax.scatter(clean[:, 0],   clean[:, 1],   c='#aec7e8', alpha=0.5, s=15, label='Clean')
        ax.scatter(outlier[:, 0], outlier[:, 1], c='#ffbbbb', alpha=0.5, s=15, label='Outlier')

        estimators = [
            (true_mean, 'k+',  'True mean',    14),
            (l2_mean,   'r x', 'l2',           12),
            (l1_mean,   'bX',  'l1',           12),
            (mcp_mean,  'g*',  'MCP (ours)',   14),
        ]
        for pt, marker_str, label, ms in estimators:
            ax.plot(pt[0], pt[1], marker_str, markersize=ms, label=label, markeredgewidth=2)

        ax.set_title(f'{int(ratio * 100)}% outliers')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_xlim(-4, 11)
        ax.set_ylim(-4, 11)

    axes[0].legend(loc='upper left', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# NEW-1: Accuracy vs attack budget (penalty comparison)
# ---------------------------------------------------------------------------

def plot_robustness_curves(
    results_csv: str,
    datasets: list,
    save_path: str = 'figures/fig_robustness.pdf',
) -> None:
    """
    Plot accuracy vs attack budget for all penalty functions.

    Reproduces the style of Figure 1 in the paper, extended with SCAD and adaptive.

    Args:
        results_csv: Path to CSV from run_ablation.py
        datasets:    List of dataset names to include as subplots
        save_path:   Output PDF path
    """
    _ensure_dir(save_path)
    df = pd.read_csv(results_csv)

    grouped = df.groupby(['dataset', 'penalty', 'budget']).agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
    ).reset_index()

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        data = grouped[grouped['dataset'] == dataset]
        penalties = sorted(data['penalty'].unique(),
                           key=lambda p: list(LABELS.keys()).index(p) if p in LABELS else 99)

        for penalty in penalties:
            pdata = data[data['penalty'] == penalty].sort_values('budget')
            if pdata.empty:
                continue
            color  = COLORS.get(penalty, 'gray')
            marker = MARKERS.get(penalty, 'o')
            label  = LABELS.get(penalty, penalty)

            ax.errorbar(
                pdata['budget'],
                pdata['accuracy_mean'] * 100,
                yerr=pdata['accuracy_std'] * 100,
                label=label, color=color, marker=marker,
                capsize=3, linewidth=1.8, markersize=6,
            )

        ax.set_title(dataset.replace('_', ' ').title())
        ax.set_xlabel('Attack Budget (% edges)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.legend(loc='lower left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6 reproduction + extension: bias vs attack budget
# ---------------------------------------------------------------------------

def plot_bias_curves(
    results_csv: str,
    save_path: str = 'figures/fig_bias_curves.pdf',
) -> None:
    """
    Reproduce and extend Figure 6: estimation bias vs attack budget.

    Shows that MCP/SCAD maintain near-zero bias while l1 bias grows with budget.
    """
    _ensure_dir(save_path)
    df = pd.read_csv(results_csv)
    df = df[df['bias_total'].notna() & (df['bias_total'] != float('nan'))]

    try:
        df['bias_total'] = df['bias_total'].astype(float)
        df = df[df['bias_total'].notna()]
    except Exception:
        print("Warning: no bias data found in CSV. Run with measure_bias=True.")
        return

    grouped = df.groupby(['dataset', 'penalty', 'budget']).agg(
        bias_mean=('bias_total', 'mean'),
        bias_std=('bias_total', 'std'),
    ).reset_index()

    datasets = grouped['dataset'].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        data = grouped[grouped['dataset'] == dataset]
        for penalty in ['L1', 'MCP', 'SCAD']:
            pdata = data[data['penalty'] == penalty].sort_values('budget')
            if pdata.empty:
                continue
            ax.plot(
                pdata['budget'], pdata['bias_mean'],
                label=LABELS.get(penalty, penalty),
                color=COLORS.get(penalty, 'gray'),
                marker=MARKERS.get(penalty, 'o'),
            )
            ax.fill_between(
                pdata['budget'],
                pdata['bias_mean'] - pdata['bias_std'].fillna(0),
                pdata['bias_mean'] + pdata['bias_std'].fillna(0),
                color=COLORS.get(penalty, 'gray'), alpha=0.15,
            )

        ax.set_title(dataset.replace('_', ' ').title())
        ax.set_xlabel('Attack Budget (% edges)')
        ax.set_ylabel('Estimation Bias  Σ||f − f*||²')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 12 reproduction: gamma vs lambda_hat heatmap
# ---------------------------------------------------------------------------

def plot_gamma_heatmap(
    results_csv: str,
    budget: int = 20,
    save_path: str = 'figures/fig_gamma_heatmap.pdf',
) -> None:
    """
    Reproduce Figure 12: accuracy heatmap over gamma × lambda_hat.

    Args:
        results_csv: Path to gamma_sensitivity experiment CSV
        budget:      Attack budget to show (default 20%)
        save_path:   Output path
    """
    _ensure_dir(save_path)
    df = pd.read_csv(results_csv)
    df = df[df['budget'] == budget]

    pivot = df.groupby(['gamma', 'lambda_hat'])['accuracy'].mean().reset_index()
    pivot = pivot.pivot(index='gamma', columns='lambda_hat', values='accuracy')

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values * 100, aspect='auto', origin='lower',
                   cmap='RdYlGn', vmin=50, vmax=85)
    plt.colorbar(im, ax=ax, label='Accuracy (%)')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v:.1f}' for v in pivot.index])
    ax.set_xlabel('λ̂ (lambda_hat)')
    ax.set_ylabel('γ (gamma)')
    ax.set_title(f'Accuracy (%) at Budget {budget}%')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 13 reproduction: performance vs number of layers
# ---------------------------------------------------------------------------

def plot_layer_sensitivity(
    results_csv: str,
    save_path: str = 'figures/fig_layer_sensitivity.pdf',
) -> None:
    """Reproduce Figure 13: accuracy vs number of propagation layers."""
    _ensure_dir(save_path)
    df = pd.read_csv(results_csv)

    grouped = df.groupby(['penalty', 'budget', 'num_layers']).agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
    ).reset_index()

    budgets = sorted(grouped['budget'].unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(4 * len(budgets), 4), sharey=True)
    if len(budgets) == 1:
        axes = [axes]

    for ax, budget in zip(axes, budgets):
        data = grouped[grouped['budget'] == budget]
        for penalty in data['penalty'].unique():
            pdata = data[data['penalty'] == penalty].sort_values('num_layers')
            ax.errorbar(
                pdata['num_layers'], pdata['accuracy_mean'] * 100,
                yerr=pdata['accuracy_std'] * 100,
                label=LABELS.get(penalty, penalty),
                color=COLORS.get(penalty, 'gray'),
                marker=MARKERS.get(penalty, 'o'), capsize=3,
            )
        ax.set_title(f'Budget {budget}%')
        ax.set_xlabel('Propagation Layers')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# NEW-2: Accuracy vs homophily ratio
# ---------------------------------------------------------------------------

def plot_homophily_vs_performance(
    results_csv: str,
    homophily_map: dict = None,
    save_path: str = 'figures/fig_homophily.pdf',
) -> None:
    """
    NEW FIGURE: Accuracy vs edge homophily ratio across datasets.

    Reveals how RUNG degrades on heterophilic graphs and whether adaptive
    penalty recovers performance.

    Args:
        results_csv:   Path to heterophilic experiment CSV
        homophily_map: Dict {dataset_name → homophily_ratio}.
                       Defaults to HOMOPHILY_MAP if None.
        save_path:     Output path
    """
    _ensure_dir(save_path)
    if homophily_map is None:
        homophily_map = HOMOPHILY_MAP

    df = pd.read_csv(results_csv)
    # Focus on clean accuracy (budget=0)
    clean_df = df[df['budget'] == 0].groupby(['dataset', 'penalty']).agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))

    for penalty in ['L1', 'MCP', 'ADAPTIVE']:
        pdata = clean_df[clean_df['penalty'] == penalty].copy()
        pdata['homophily'] = pdata['dataset'].map(homophily_map)
        pdata = pdata.dropna(subset=['homophily']).sort_values('homophily')
        if pdata.empty:
            continue

        color  = COLORS.get(penalty, 'gray')
        marker = MARKERS.get(penalty, 'o')
        label  = LABELS.get(penalty, penalty)

        ax.errorbar(
            pdata['homophily'], pdata['accuracy_mean'] * 100,
            yerr=pdata['accuracy_std'] * 100,
            label=label, color=color, marker=marker,
            capsize=3, linewidth=1.8, markersize=8,
        )
        # Annotate dataset names
        for _, row in pdata.iterrows():
            ax.annotate(
                row['dataset'],
                (row['homophily'], row['accuracy_mean'] * 100),
                textcoords='offset points', xytext=(4, 4),
                fontsize=8, color=color, alpha=0.8,
            )

    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4, label='h=0.5 (threshold)')
    ax.set_xlabel('Edge Homophily Ratio h')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title('Accuracy vs Graph Homophily')
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# NEW-3: Bar chart — clean accuracy across penalties
# ---------------------------------------------------------------------------

def plot_penalty_comparison_bars(
    results_csv: str,
    dataset: str = 'cora',
    save_path: str = 'figures/fig_penalty_bars.pdf',
) -> None:
    """
    Bar chart comparing clean accuracy across all penalty functions.

    Args:
        results_csv: Path to penalty_comparison experiment CSV
        dataset:     Dataset to plot
        save_path:   Output path
    """
    _ensure_dir(save_path)
    df = pd.read_csv(results_csv)
    df = df[(df['dataset'] == dataset) & (df['budget'] == 0)]

    summary = df.groupby('penalty').agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(summary))
    colors = [COLORS.get(p, 'gray') for p in summary['penalty']]
    labels = [LABELS.get(p, p) for p in summary['penalty']]

    bars = ax.bar(x, summary['accuracy_mean'] * 100,
                  yerr=summary['accuracy_std'] * 100,
                  color=colors, edgecolor='white', capsize=4, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title(f'Penalty Comparison — {dataset.title()} (budget=0%)')
    ax.set_ylim(0, 100)

    # Annotate values
    for bar, val in zip(bars, summary['accuracy_mean']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val*100:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot RUNG experiment results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results_csv', type=str, default=None,
        help='Path to CSV file from run_ablation.py (required for most figures)',
    )
    parser.add_argument(
        '--figure', type=str, required=True,
        choices=[
            'bias_simulation',
            'robustness',
            'bias_curves',
            'gamma_heatmap',
            'layer_sensitivity',
            'homophily',
            'penalty_bars',
        ],
        help='Which figure to generate',
    )
    parser.add_argument(
        '--save_dir', type=str, default='./figures',
        help='Directory to save the output figure',
    )
    parser.add_argument(
        '--dataset', type=str, default='cora',
        help='Dataset filter for figures that show a single dataset',
    )
    parser.add_argument(
        '--budget', type=int, default=20,
        help='Attack budget filter for gamma_heatmap',
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    sdir = args.save_dir

    if args.figure == 'bias_simulation':
        plot_bias_simulation(f'{sdir}/fig_bias_simulation.pdf')

    elif args.figure == 'robustness':
        if not args.results_csv:
            parser.error("--results_csv is required for --figure robustness")
        plot_robustness_curves(
            args.results_csv,
            datasets=['cora', 'citeseer'],
            save_path=f'{sdir}/fig_robustness.pdf',
        )

    elif args.figure == 'bias_curves':
        if not args.results_csv:
            parser.error("--results_csv is required for --figure bias_curves")
        plot_bias_curves(args.results_csv, save_path=f'{sdir}/fig_bias_curves.pdf')

    elif args.figure == 'gamma_heatmap':
        if not args.results_csv:
            parser.error("--results_csv is required for --figure gamma_heatmap")
        plot_gamma_heatmap(
            args.results_csv, budget=args.budget,
            save_path=f'{sdir}/fig_gamma_heatmap.pdf',
        )

    elif args.figure == 'layer_sensitivity':
        if not args.results_csv:
            parser.error("--results_csv is required for --figure layer_sensitivity")
        plot_layer_sensitivity(args.results_csv, save_path=f'{sdir}/fig_layer_sensitivity.pdf')

    elif args.figure == 'homophily':
        if not args.results_csv:
            parser.error("--results_csv is required for --figure homophily")
        plot_homophily_vs_performance(
            args.results_csv, save_path=f'{sdir}/fig_homophily.pdf'
        )

    elif args.figure == 'penalty_bars':
        if not args.results_csv:
            parser.error("--results_csv is required for --figure penalty_bars")
        plot_penalty_comparison_bars(
            args.results_csv, dataset=args.dataset,
            save_path=f'{sdir}/fig_penalty_bars_{args.dataset}.pdf',
        )
