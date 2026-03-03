#!/usr/bin/env python
"""
experiments/analyze_learned_gammas.py
======================================
Analyses what RUNG_learnable_gamma learned about per-layer SCAD thresholds.

Produces:
1. Gamma convergence plot — gamma value vs training epoch, one line per layer.
2. Feature-difference distribution plot — histogram of y_ij at each layer,
   with the learned gamma overlaid as vertical markers.
3. Summary table comparing RUNG_new_SCAD (fixed gamma) vs RUNG_learnable_gamma.

Usage (standalone demonstration on Cora, requires a trained model checkpoint
or runs a quick training run for demonstration):

    python experiments/analyze_learned_gammas.py

Usage from within Python (passing a pre-trained model):

    from experiments.analyze_learned_gammas import (
        plot_gamma_convergence,
        plot_feature_diff_per_layer,
    )

    model, gamma_history = ...   # from your training run
    plot_gamma_convergence(gamma_history)
    plot_feature_diff_per_layer(model, A, X)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pathlib import Path


# ============================================================
#  1.  Gamma convergence plot
# ============================================================

def plot_gamma_convergence(
    gamma_history: dict,
    scad_a: float = 3.7,
    save_path: str = "figures/gamma_convergence.pdf",
) -> None:
    """
    Plot learned gamma (= a × lam = SCAD zero-cutoff) values over training epochs.

    One line per layer, coloured by depth (viridis colour map).
    Helps answer: "do later layers converge to smaller gammas?"

    Args:
        gamma_history: dict mapping epoch (int) → list of gamma values per layer.
                       Produced by fit_learnable_gamma when log_gamma_every > 0.
                       Format: {50: [g0, g1, ...], 100: [g0, g1, ...], ...}
        scad_a:        SCAD shape param (for axis labelling only).
        save_path:     Output file path (PDF or PNG).
    """
    if not gamma_history:
        print("gamma_history is empty — nothing to plot.")
        return

    epochs     = sorted(gamma_history.keys())
    num_layers = len(gamma_history[epochs[0]])

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [cm.viridis(k / max(num_layers - 1, 1)) for k in range(num_layers)]

    for k in range(num_layers):
        gammas_k = [gamma_history[e][k] for e in epochs]
        ax.plot(
            epochs, gammas_k,
            color=colors[k],
            linewidth=1.8,
            label=f"Layer {k}",
        )

    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel(r"Learned $\gamma^{(k)}$ (SCAD zero-cutoff = $a \cdot \lambda^{(k)}$)", fontsize=12)
    ax.set_title("Per-Layer Gamma Convergence\nRUNG_learnable_gamma", fontsize=13)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        fontsize=8,
        title="Layer",
    )
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved gamma convergence plot: {save_path}")
    plt.close()


# ============================================================
#  2.  Feature-difference distribution per layer
# ============================================================

def plot_feature_diff_per_layer(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    save_path: str = "figures/feature_diffs_per_layer.pdf",
    ncols: int = 5,
) -> None:
    """
    Plot histogram of edge feature differences y_ij at each propagation layer.

    Overlays:
        Red vertical line   = learned gamma^(k)  (SCAD pruning starts here)
        Orange dashed line  = a * gamma^(k)       (SCAD pruning ends here)

    Expected pattern: distributions shift left (smaller diffs) as depth
    increases, and learned gammas shift left correspondingly.

    Args:
        model:     Trained RUNG_learnable_gamma instance.
        A:         [N, N] adjacency matrix (dense, with self-loops already or without).
        X:         [N, D] node feature matrix.
        save_path: Output file path.
        ncols:     Number of subplot columns. Rows are computed automatically.
    """
    from utils import add_loops, pairwise_squared_euclidean, sym_norm

    K = model.prop_layer_num
    nrows = (K + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes_flat = axes.flatten() if nrows > 1 else axes

    model.eval()
    device = next(model.parameters()).device
    A = A.to(device)
    X = X.to(device)

    with torch.no_grad():
        A_loop  = add_loops(A)
        D       = A_loop.sum(-1)
        D_sq    = D.sqrt().unsqueeze(-1)
        A_tilde = sym_norm(A_loop)

        F0 = model.mlp(X)
        F  = F0.clone()

        for k, log_lam_param in enumerate(model.log_lams):
            ax = axes_flat[k]

            # Compute y_ij at this layer
            Z  = pairwise_squared_euclidean(F / D_sq, F / D_sq)
            y  = Z.sqrt()

            # Extract only off-diagonal (actual edges in A)
            mask        = A > 0
            y_edges     = y[mask].cpu().numpy()

            lam_k = float(torch.exp(log_lam_param).item())
            gamma_k = model.scad_a * lam_k     # zero-cutoff

            # Histogram
            ax.hist(y_edges, bins=60, density=True, color="steelblue", alpha=0.75)

            ax.axvline(x=lam_k,   color="red",    linewidth=2.0,
                       label=rf"$\lambda$={lam_k:.2f}")
            ax.axvline(x=gamma_k, color="darkorange", linewidth=1.8,
                       linestyle="--",
                       label=rf"$a\lambda$={gamma_k:.2f}")

            ax.set_title(f"Layer {k}", fontsize=10)
            ax.set_xlabel(r"$y_{ij}$", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

            # Propagate to next layer for next iteration
            lam_k_t  = torch.exp(log_lam_param)
            from model.rung_learnable_gamma import scad_weight_differentiable
            W        = scad_weight_differentiable(y, lam_k_t, a=model.scad_a)
            idx      = torch.arange(W.shape[0], device=device)
            W[idx, idx] = 0.0
            W[torch.isnan(W)]  = 1.0
            Q_hat    = ((W * A_loop).sum(-1) / D + model.lam).unsqueeze(-1)
            F        = (W * A_tilde) @ F / Q_hat + model.lam * F0 / Q_hat

    # Hide unused subplots
    for i in range(K, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        "Feature Difference Distributions per Layer\n"
        r"Red $\lambda^{(k)}$ = SCAD threshold,  Orange $a\cdot\lambda^{(k)}$ = zero-cutoff",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved feature diff plot: {save_path}")
    plt.close()


# ============================================================
#  3.  Summary comparison table
# ============================================================

def print_comparison_table(
    results_fixed: dict,
    results_learnable: dict,
) -> None:
    """
    Print a markdown-formatted comparison table.

    Args:
        results_fixed:     {budget: (clean_acc, attacked_acc)}  for RUNG_new_SCAD.
        results_learnable: {budget: (clean_acc, attacked_acc)}  for RUNG_learnable_gamma.
    """
    budgets = sorted(set(list(results_fixed.keys()) + list(results_learnable.keys())))

    print(f"\n{'='*72}")
    print(f"{'Comparison: RUNG_new_SCAD vs RUNG_learnable_gamma':^72}")
    print(f"{'='*72}")
    print(
        f"{'Budget':>8} | "
        f"{'RUNG_new_SCAD clean':>20} | "
        f"{'RUNG_new_SCAD atk':>18} | "
        f"{'LearGamma clean':>16} | "
        f"{'LearGamma atk':>14}"
    )
    print(f"{'-'*72}")
    for b in budgets:
        fc, fa = results_fixed.get(b,     ("—", "—"))
        lc, la = results_learnable.get(b, ("—", "—"))
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(
            f"{b:>8.3f} | {fmt(fc):>20} | {fmt(fa):>18} | "
            f"{fmt(lc):>16} | {fmt(la):>14}"
        )
    print(f"{'='*72}\n")


# ============================================================
#  4.  Quick demonstration / self-test
# ============================================================

if __name__ == "__main__":
    import argparse
    import copy

    parser = argparse.ArgumentParser(description="Analyze RUNG_learnable_gamma gamma values")
    parser.add_argument("--data",   type=str,   default="cora")
    parser.add_argument("--gamma",  type=float, default=6.0)
    parser.add_argument("--epochs", type=int,   default=300)
    parser.add_argument("--gamma_lr_factor", type=float, default=0.2)
    parser.add_argument(
        "--gamma_init_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "decreasing", "increasing"],
    )
    parser.add_argument(
        "--save_dir", type=str, default="figures",
        help="Directory for output figures"
    )
    args = parser.parse_args()

    # ------- imports -------
    from train_eval_data.get_dataset import get_dataset, get_splits
    from exp.config.get_model import get_model_default
    from train_eval_data.fit_learnable_gamma import fit_learnable_gamma
    from utils import accuracy

    print(f"Dataset: {args.data}  |  gamma_init={args.gamma}  |  "
          f"strategy={args.gamma_init_strategy}")

    # ------- build model -------
    model, fit_params = get_model_default(
        args.data,
        "RUNG_learnable_gamma",
        custom_model_params={
            "gamma": args.gamma,
            "gamma_init_strategy": args.gamma_init_strategy,
        },
        custom_fit_params={"lr": 5e-2, "weight_decay": 5e-4, "max_epoch": args.epochs},
    )

    A, X, y     = get_dataset(args.data)
    splits      = get_splits(y)
    train_idx, val_idx, test_idx = splits[0]

    device = next(model.parameters()).device
    A = A.to(device); X = X.to(device); y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx   = val_idx.to(device)
    test_idx  = test_idx.to(device)

    print(model)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Gamma params: {sum(p.numel() for p in model.get_gamma_parameters()):,}")

    # ------- train with gamma logging -------
    gamma_history = {}

    from train_eval_data.fit_learnable_gamma import (
        build_two_group_optimizer,
    )
    import torch.nn.functional as F_nn

    optimizer = build_two_group_optimizer(
        model, lr=5e-2, gamma_lr_factor=args.gamma_lr_factor
    )
    best_val, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(A, X)
        loss   = F_nn.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            va = accuracy(model(A, X)[val_idx], y[val_idx]).item()
        if va > best_val:
            best_val  = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            gamma_history[epoch] = model.get_learned_gammas()
            print(f"Epoch {epoch:4d} | loss={loss.item():.4f} | val={va:.4f} | "
                  f"gammas=[{', '.join(f'{g:.2f}' for g in gamma_history[epoch])}]")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_acc = accuracy(model(A, X)[test_idx], y[test_idx]).item()
    print(f"\nTest accuracy: {test_acc:.4f}")

    model.log_gamma_stats()

    # ------- plots -------
    os.makedirs(args.save_dir, exist_ok=True)

    if gamma_history:
        plot_gamma_convergence(
            gamma_history,
            save_path=os.path.join(args.save_dir, "gamma_convergence.pdf"),
        )

    plot_feature_diff_per_layer(
        model, A, X,
        save_path=os.path.join(args.save_dir, "feature_diffs_per_layer.pdf"),
    )

    print("\nDone. Figures saved to:", args.save_dir)
