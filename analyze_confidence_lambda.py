"""
analyze_confidence_lambda.py

Analysis specific to RUNG_confidence_lambda.
Answers the key research questions:

Q1: Does the model actually redistribute lambda as intended?
    -> plot_lambda_vs_confidence()

Q2: Are attacked nodes actually less confident (validating the hypothesis)?
    -> plot_confidence_under_attack()

Q3: How does the lambda distribution change as attack budget increases?
    -> plot_lambda_distribution_vs_budget()

Q4: Does alpha converge to a meaningful value during training?
    -> plot_alpha_convergence()

Usage example:
    python analyze_confidence_lambda.py

    Or import and call individual functions:
        from analyze_confidence_lambda import plot_lambda_vs_confidence
        plot_lambda_vs_confidence(model, A, X, y)
"""

import sys
sys.path.insert(0, ".")

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe in all environments)
import matplotlib.pyplot as plt

from model.rung_confidence_lambda import RUNG_confidence_lambda, compute_confidence_lambda
from train_eval_data.get_dataset import get_dataset, get_splits


os.makedirs('figures/confidence_lambda', exist_ok=True)


# ============================================================
# Q1: Lambda vs Confidence scatter
# ============================================================

def plot_lambda_vs_confidence(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor = None,
    save_path: str = 'figures/confidence_lambda/lambda_vs_confidence.pdf',
):
    """
    Scatter plot of per-node lambda vs prediction confidence.

    Expected patterns by mode:
        'protect_uncertain'  → negative slope  (uncertain → higher lambda)
        'protect_confident'  → positive slope  (confident → higher lambda)
        'symmetric'          → arch shape      (mid-confidence → highest lambda)

    Optional color-coding by true label shows whether any class is
    systematically more or less confident under the current graph.

    Args:
        model:     Trained RUNG_confidence_lambda (eval mode).
        A:         [N, N] adjacency (clean or perturbed).
        X:         [N, D] node features.
        y:         [N] integer class labels (optional; enables color-coding).
        save_path: Output path (PDF or PNG).
    """
    lambda_vals, conf_vals, summary = model.get_lambda_distribution(A, X)

    x_np  = conf_vals.numpy()
    y_np  = lambda_vals.numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap    = plt.cm.tab10

    if y is not None:
        labels      = y.cpu().numpy()
        num_classes = int(labels.max()) + 1
        for c in range(num_classes):
            mask = labels == c
            ax.scatter(
                x_np[mask], y_np[mask],
                c=[cmap(c / num_classes)],
                alpha=0.4, s=10,
                label=f'Class {c}',
            )
    else:
        ax.scatter(x_np, y_np, alpha=0.4, s=10, color='steelblue')

    # Linear regression trend line
    z      = np.polyfit(x_np, y_np, 1)
    p      = np.poly1d(z)
    xline  = np.linspace(x_np.min(), x_np.max(), 100)
    ax.plot(xline, p(xline), 'k--', linewidth=2,
            label=f'Trend (slope={z[0]:.3f})')

    # Reference line at lambda_base
    ax.axhline(
        y=summary['lambda_base'], color='gray',
        linestyle=':', linewidth=1.5,
        label=f"λ_base={summary['lambda_base']:.3f}",
    )

    corr = float(np.corrcoef(x_np, y_np)[0, 1])
    ax.set_xlabel('Prediction Confidence (max softmax)')
    ax.set_ylabel('Per-node Lambda λ_i')
    ax.set_title(
        f"Lambda vs Confidence\n"
        f"mode={summary['mode']},  α={summary['alpha']:.3f},  "
        f"Pearson r={corr:.3f}"
    )
    if y is not None:
        ax.legend(loc='upper right', fontsize=7, ncol=2)
    else:
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return corr


# ============================================================
# Q2: Confidence distribution under increasing attack
# ============================================================

def plot_confidence_under_attack(
    model,
    A_clean: torch.Tensor,
    X: torch.Tensor,
    A_perturbed_list: list,
    budgets: list,
    save_path: str = 'figures/confidence_lambda/confidence_under_attack.pdf',
):
    """
    Show how node confidence distribution shifts under increasing attack.

    Key hypothesis: under attack, the model's pre-aggregation confidence
    decreases on average (adversarial edges add noise near decision boundaries
    → harder to be confident from features alone).

    If this is true, the confidence-lambda mechanism is responding to the
    right signal: lambda is redistributed exactly when neighbourhoods become
    corrupted.

    Args:
        model:               Trained RUNG_confidence_lambda (eval).
        A_clean:             [N, N] clean adjacency.
        X:                   [N, D] node features.
        A_perturbed_list:    list of [N, N] adjacency matrices, one per budget.
        budgets:             list of budget values (for subplot titles).
        save_path:           Output path.
    """
    all_A      = [A_clean] + A_perturbed_list
    all_labels = ['Clean (budget=0)'] + [f'Budget {b}' for b in budgets]
    n_plots    = len(all_A)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]

    model.eval()
    for ax, A_cur, label in zip(axes, all_A, all_labels):
        with torch.no_grad():
            F0    = model.mlp(X)
            probs = torch.softmax(F0, dim=-1)
            conf  = probs.max(dim=-1).values.cpu().numpy()

        ax.hist(conf, bins=30, density=True, color='steelblue', alpha=0.7)
        ax.axvline(x=float(conf.mean()), color='red', linewidth=2,
                   label=f'mean={conf.mean():.3f}')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Confidence')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Density')
    plt.suptitle(
        'Node Confidence Distribution Under Increasing Attack\n'
        '(If hypothesis correct: distribution shifts left with budget)',
        y=1.02, fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Q3: Lambda distribution vs attack budget
# ============================================================

def plot_lambda_distribution_vs_budget(
    model,
    A_clean: torch.Tensor,
    X: torch.Tensor,
    A_perturbed_list: list,
    budgets: list,
    save_path: str = 'figures/confidence_lambda/lambda_vs_budget.pdf',
):
    """
    Show how the per-node lambda distribution shifts under increasing attack.

    For 'protect_uncertain' mode, as the attack causes more nodes to become
    uncertain, the lambda distribution should shift right (more high-lambda
    nodes = more nodes strongly relying on own features).

    Args:
        model:               Trained RUNG_confidence_lambda (eval).
        A_clean:             [N, N] clean adjacency.
        X:                   [N, D] node features.
        A_perturbed_list:    list of [N, N] adjacency matrices.
        budgets:             list of budget labels.
        save_path:           Output path.
    """
    all_A      = [A_clean] + A_perturbed_list
    all_labels = ['Clean'] + [f'Budget {b}' for b in budgets]
    n_plots    = len(all_A)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]

    model.eval()
    for ax, A_cur, label in zip(axes, all_A, all_labels):
        lambda_vals, conf_vals, summary = model.get_lambda_distribution(A_cur, X)
        lam_np = lambda_vals.numpy()

        ax.hist(lam_np, bins=30, density=True, color='mediumpurple', alpha=0.7)
        ax.axvline(x=float(lam_np.mean()), color='red', linewidth=2,
                   label=f'mean={lam_np.mean():.3f}')
        ax.axvline(x=summary['lambda_base'], color='gray', linewidth=1.5,
                   linestyle='--', label=f"base={summary['lambda_base']:.3f}")
        ax.set_title(f'{label}\nstd={lam_np.std():.4f}', fontsize=9)
        ax.set_xlabel('Lambda λ_i')
        ax.legend(fontsize=7)

    axes[0].set_ylabel('Density')
    plt.suptitle(
        f'Per-node Lambda Distribution vs Attack Budget\n'
        f"mode='{model.confidence_mode}', α={model.alpha:.3f}",
        y=1.02, fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Q4: Alpha convergence during training
# ============================================================

def plot_alpha_convergence(
    history: dict,
    save_path: str = 'figures/confidence_lambda/alpha_convergence.pdf',
):
    """
    Plot how alpha and lambda statistics evolve during training.

    Three subplots:
        Left:   alpha value per log interval.
        Middle: std(lambda_i) — non-zero means redistribution is active.
        Right:  Pearson corr(lambda, confidence) — reveals mode-specific patterns.

    Expected healthy convergence:
        alpha  → stable value in (0.5, 5.0)
        λ_std  → non-zero after warmup ends
        corr   → negative for protect_uncertain, positive for protect_confident

    Args:
        history:   dict from fit_confidence_lambda (or train_model) containing
                   keys 'alpha', 'lambda_std', 'lambda_conf_corr'.
        save_path: Output path.
    """
    if not history.get('alpha'):
        print("No alpha history available. Skipping convergence plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Alpha value
    axes[0].plot(history['alpha'], color='purple', linewidth=2)
    axes[0].axhline(y=1.0, color='gray', linestyle='--',
                    label='α=1 (linear map)', linewidth=1.5)
    axes[0].axhline(y=0.5, color='red', linestyle=':', linewidth=1,
                    label='α=0.5 (floor)')
    axes[0].set_xlabel('Log interval')
    axes[0].set_ylabel('Alpha value')
    axes[0].set_title('Alpha Convergence\n(healthy: settles between 0.5–5)')
    axes[0].legend(fontsize=8)

    # Lambda std
    if history.get('lambda_std'):
        axes[1].plot(history['lambda_std'], color='green', linewidth=2)
        axes[1].set_xlabel('Log interval')
        axes[1].set_ylabel('std(λ_i)')
        axes[1].set_title(
            'Lambda Redistribution (std)\n'
            '(healthy: non-zero after warmup ends)'
        )

    # Lambda-confidence correlation
    if history.get('lambda_conf_corr'):
        axes[2].plot(history['lambda_conf_corr'], color='orange', linewidth=2)
        axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
        axes[2].set_xlabel('Log interval')
        axes[2].set_ylabel('Pearson r(λ, confidence)')
        axes[2].set_title(
            'Lambda–Confidence Correlation\n'
            '(protect_uncertain: negative; protect_confident: positive)'
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Convenience: quick summary for a trained model
# ============================================================

def print_model_summary(model, A: torch.Tensor, X: torch.Tensor) -> None:
    """
    Print a concise summary of the trained model's lambda distribution
    and learned parameters.

    Args:
        model: Trained RUNG_confidence_lambda.
        A:     [N, N] adjacency matrix.
        X:     [N, D] node features.
    """
    model.log_gamma_stats()

    _, _, summary = model.get_lambda_distribution(A, X)

    print("Lambda distribution on provided graph:")
    print(f"  mean:      {summary['lambda_mean']:.4f}  (target: {summary['lambda_base']:.4f})")
    print(f"  std:       {summary['lambda_std']:.4f}")
    print(f"  range:     [{summary['lambda_min']:.4f}, {summary['lambda_max']:.4f}]")
    print(f"  conf mean: {summary['conf_mean']:.4f}  ±  {summary['conf_std']:.4f}")
    print(f"  alpha:     {summary['alpha']:.4f}")
    print(f"  mode:      {summary['mode']}")


# ============================================================
# Demo / quick-sanity run
# ============================================================

if __name__ == '__main__':
    """
    Quick sanity demo using random data.
    Replace with a real trained model and dataset for actual analysis.
    """
    import torch
    from model.rung_confidence_lambda import RUNG_confidence_lambda

    torch.manual_seed(42)
    N, D, C = 300, 64, 7

    # Random graph (Erdos-Renyi style)
    A_demo = (torch.rand(N, N) > 0.95).float()
    A_demo = (A_demo + A_demo.t()).clamp(max=1.0)
    A_demo.fill_diagonal_(0.0)

    X_demo = torch.randn(N, D)
    y_demo = torch.randint(0, C, (N,))

    print("Creating demo RUNG_confidence_lambda model...")
    model_demo = RUNG_confidence_lambda(
        in_dim=D, out_dim=C, hidden_dims=[64],
        lam_hat=0.9, gamma_init=6.0, prop_step=3,
        confidence_mode='protect_uncertain',
        normalize_lambda=True,
    )
    model_demo.eval()

    print("\nRunning Q1: Lambda vs Confidence plot...")
    corr = plot_lambda_vs_confidence(
        model_demo, A_demo, X_demo, y_demo,
        save_path='figures/confidence_lambda/demo_lambda_vs_confidence.pdf',
    )
    print(f"  Pearson r = {corr:.4f}")

    print("\nRunning Q2: Confidence under attack plot (demo with clean data × 3)...")
    plot_confidence_under_attack(
        model_demo, A_demo, X_demo,
        A_perturbed_list=[A_demo, A_demo],
        budgets=[10, 20],
        save_path='figures/confidence_lambda/demo_confidence_under_attack.pdf',
    )

    print("\nRunning model summary...")
    print_model_summary(model_demo, A_demo, X_demo)

    print("\nDemo complete. See figures/confidence_lambda/ for output plots.")
