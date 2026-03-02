#!/usr/bin/env python
"""
experiments/compare_models.py
==============================
Compare RUNG (original baseline) vs RUNG_new (refactored, penalty.py path)
and standard GNN baselines, all in one run.

Run directly:
    python experiments/compare_models.py

Or from run_all.py:
    python run_all.py --compare

Results are written to:
    results/comparison/comparison_<timestamp>.csv

Figures are written to:
    figures/comparison/
"""

import sys
import os
import subprocess
import time
import csv
import yaml
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "comparison"
FIGURES_DIR = ROOT / "figures" / "comparison"

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
# (display_label, --model arg, --norm arg, --gamma arg)
DEFAULT_MODEL_CONFIGS = [
    ("RUNG",          "RUNG",     "MCP",   6.0),   # original — att_func.py code path
    ("RUNG_new",      "RUNG_new", "MCP",   6.0),   # refactored — penalty.py code path
    ("RUNG_new_SCAD", "RUNG_new", "SCAD",  6.0),   # new SCAD penalty (docs/001)
    ("GCN",           "GCN",      "MCP",   6.0),   # GCN baseline
    ("MLP",           "MLP",      "MCP",   6.0),   # no-propagation baseline
]

# Attack budgets (fraction of total edges)
ATTACK_BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.40]

# ---------------------------------------------------------------------------
# Color scheme (from docs/006_visualization.md)
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "RUNG":          "#2ca02c",   # green  — MCP / original RUNG
    "RUNG_new":      "#17becf",   # cyan   — new MCP implementation
    "RUNG_new_SCAD": "#ff7f0e",   # orange — SCAD
    "GCN":           "#d62728",   # red    — GCN
    "MLP":           "#9467bd",   # purple — MLP
}
MODEL_MARKERS = {
    "RUNG":          "o",
    "RUNG_new":      "s",
    "RUNG_new_SCAD": "^",
    "GCN":           "D",
    "MLP":           "x",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PYTHON = sys.executable


def _run_subprocess(script: str, model: str, norm: str, gamma: float,
                    dataset: str, extra_args: list = None,
                    max_epoch: int = 300) -> tuple:
    """Invoke clean.py or attack.py for one (model, dataset) config."""
    cmd = [
        PYTHON, str(ROOT / script),
        f"--model={model}",
        f"--norm={norm}",
        f"--gamma={gamma}",
        f"--data={dataset}",
    ]
    if script == "clean.py":
        cmd.append(f"--max_epoch={max_epoch}")
    if extra_args:
        cmd.extend(extra_args)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    elapsed = time.perf_counter() - t0
    return result.returncode == 0, elapsed


def _read_clean_result(dataset: str, model: str, gamma: float) -> dict | None:
    """Read clean accuracy from the YAML saved by clean.py."""
    fname = ROOT / f"exp/result/{dataset}/clean_{model}_{gamma}.yaml"
    if not fname.exists():
        return None
    with open(fname) as f:
        data = yaml.safe_load(f)
    if data is None:
        return None
    # The YAML has structure: {model_name: {result: {mean, std, accs, ...}}}
    # but model_name is written by clean.py using args.model (e.g. "RUNG")
    # Try to find any top-level key that has "result"
    for key, val in data.items():
        if isinstance(val, dict) and "result" in val:
            return val["result"]
    return None


def _read_attack_result(dataset: str, model: str, gamma: float,
                        budget_pct: int) -> dict | None:
    """Read attack accuracy from the YAML saved by attack.py."""
    fname = ROOT / (
        f"exp/result/{dataset}/"
        f"global_evasion_pgd_adaptive_{model}_{gamma}_{budget_pct}_percent.yaml"
    )
    if not fname.exists():
        return None
    with open(fname) as f:
        data = yaml.safe_load(f)
    if data is None:
        return None
    for key, val in data.items():
        if isinstance(val, dict) and "result" in val:
            return val["result"]
    return None


def _header(text: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {text}\n{bar}\n", flush=True)


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    datasets: list,
    model_configs: list = None,
    attack_budgets: list = None,
    skip_clean: bool = False,
    skip_attack: bool = False,
    max_epoch: int = 300,
    skip_plot: bool = False,
) -> Path:
    """
    Run full comparison: train + attack for every (model, dataset) pair,
    aggregate results into a CSV, and generate matplotlib figures.

    Returns the path to the CSV file.
    """
    if model_configs is None:
        model_configs = DEFAULT_MODEL_CONFIGS
    if attack_budgets is None:
        attack_budgets = ATTACK_BUDGETS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"comparison_{timestamp}.csv"
    csv_latest = RESULTS_DIR / "comparison_latest.csv"

    # -----------------------------------------------------------------------
    # Phase 1: Training
    # -----------------------------------------------------------------------
    if not skip_clean:
        _header(f"Phase 1 — Training  ({len(model_configs)} models × {len(datasets)} datasets)")
        for label, model, norm, gamma in model_configs:
            for dataset in datasets:
                print(f">>> Train  label={label}  model={model}  norm={norm}  "
                      f"gamma={gamma}  dataset={dataset}")
                ok, elapsed = _run_subprocess(
                    "clean.py", model, norm, gamma, dataset,
                    max_epoch=max_epoch
                )
                print(f"<<< {'DONE' if ok else 'FAILED'}  ({elapsed:.1f}s)\n", flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: Attack evaluation
    # -----------------------------------------------------------------------
    if not skip_attack:
        _header(f"Phase 2 — Attack  ({len(model_configs)} models × "
                f"{len(datasets)} datasets × {len(attack_budgets)} budgets)")
        for label, model, norm, gamma in model_configs:
            for dataset in datasets:
                print(f">>> Attack  label={label}  dataset={dataset}")
                ok, elapsed = _run_subprocess(
                    "attack.py", model, norm, gamma, dataset
                )
                print(f"<<< {'DONE' if ok else 'FAILED'}  ({elapsed:.1f}s)\n", flush=True)

    # -----------------------------------------------------------------------
    # Phase 3: Collect results into CSV
    # -----------------------------------------------------------------------
    _header("Phase 3 — Collecting results")
    rows = []
    for label, model, norm, gamma in model_configs:
        for dataset in datasets:
            # Clean accuracy
            clean_res = _read_clean_result(dataset, model, gamma)
            clean_mean = clean_res["mean"] if clean_res else float("nan")
            clean_std  = clean_res["std"]  if clean_res else float("nan")

            # Budget = 0 row
            rows.append({
                "label":    label,
                "model":    model,
                "norm":     norm,
                "gamma":    gamma,
                "dataset":  dataset,
                "budget":   0.0,
                "clean_acc": clean_mean,
                "clean_std": clean_std,
                "attacked_acc": clean_mean,
                "attacked_std": clean_std,
            })
            print(f"  {label:<18}  {dataset:<10}  budget=0.00  "
                  f"clean={clean_mean:.4f}±{clean_std:.4f}")

            # Attack budgets
            for bud in attack_budgets:
                bud_pct = int(round(bud * 100))
                atk_res = _read_attack_result(dataset, model, gamma, bud_pct)
                atk_mean = atk_res["mean-adv"] if atk_res else float("nan")
                atk_std  = atk_res["std-adv"]  if atk_res else float("nan")
                rows.append({
                    "label":    label,
                    "model":    model,
                    "norm":     norm,
                    "gamma":    gamma,
                    "dataset":  dataset,
                    "budget":   bud,
                    "clean_acc": clean_mean,
                    "clean_std": clean_std,
                    "attacked_acc": atk_mean,
                    "attacked_std": atk_std,
                })
                print(f"  {label:<18}  {dataset:<10}  budget={bud:.2f}  "
                      f"attacked={atk_mean:.4f}±{atk_std:.4f}")

    # Write CSV
    fieldnames = ["label", "model", "norm", "gamma", "dataset", "budget",
                  "clean_acc", "clean_std", "attacked_acc", "attacked_std"]
    for path_ in (csv_path, csv_latest):
        with open(path_, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n  Results saved to:\n    {csv_path}\n    {csv_latest}")

    # -----------------------------------------------------------------------
    # Phase 4: Generate figures
    # -----------------------------------------------------------------------
    if not skip_plot:
        _header("Phase 4 — Generating figures")
        _generate_plots(rows, datasets, model_configs, attack_budgets, timestamp)

    return csv_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _generate_plots(rows: list, datasets: list, model_configs: list,
                    attack_budgets: list, timestamp: str) -> None:
    """Generate all comparison figures and save to FIGURES_DIR."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  WARNING: matplotlib not found — skipping plots.")
        return

    # Consistent style -------------------------------------------------------
    plt.rcParams.update({
        "font.family":   "serif",
        "font.size":     11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":    150,
    })

    labels = [cfg[0] for cfg in model_configs]

    # Helper: extract data matrix ------------------------------------------------
    def _get_matrix(metric: str, budget: float, dataset: str) -> dict:
        """Return {label: (mean, std)} for a given (metric, budget, dataset)."""
        out = {}
        for row in rows:
            if row["dataset"] == dataset and abs(row["budget"] - budget) < 1e-6:
                out[row["label"]] = (row[metric], row[f"{metric.split('_')[0]}_std"])
        return out

    # ── Figure 1: Clean accuracy bar chart ──────────────────────────────────
    n_ds = len(datasets)
    n_models = len(model_configs)
    figw = max(8, n_ds * n_models * 0.6 + 1)
    fig, ax = plt.subplots(figsize=(figw, 5))
    bar_width = 0.8 / n_models
    x = np.arange(n_ds)

    for i, (label, model, norm, gamma) in enumerate(model_configs):
        means, stds = [], []
        for ds in datasets:
            d = _get_matrix("clean_acc", 0.0, ds)
            entry = d.get(label, (float("nan"), float("nan")))
            means.append(entry[0])
            stds.append(entry[1])
        color = MODEL_COLORS.get(label, f"C{i}")
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width,
               yerr=stds, capsize=3, color=color, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Clean Test Accuracy")
    ax.set_title("Clean Accuracy — RUNG vs RUNG_new vs Baselines")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save_path = FIGURES_DIR / f"clean_accuracy_{timestamp}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")

    # ── Figure 2: Accuracy vs attack budget (one subplot per dataset) ────────
    budgets_all = [0.0] + attack_budgets
    fig2, axes = plt.subplots(1, n_ds, figsize=(max(6, 4.5*n_ds), 4.5),
                              sharey=True)
    if n_ds == 1:
        axes = [axes]

    for ax2, ds in zip(axes, datasets):
        for label, model, norm, gamma in model_configs:
            means, stds = [], []
            for bud in budgets_all:
                d = _get_matrix("attacked_acc", bud, ds)
                entry = d.get(label, (float("nan"), float("nan")))
                means.append(entry[0] if not np.isnan(entry[0]) else None)
                stds.append(entry[1] if stds is not None and not np.isnan(entry[1]) else 0.0)

            # Filter None values for plot
            valid_idx = [i for i, m in enumerate(means) if m is not None]
            if not valid_idx:
                continue
            xs    = [budgets_all[i] for i in valid_idx]
            ys    = [means[i]       for i in valid_idx]
            errs  = [stds[i]        for i in valid_idx]

            color  = MODEL_COLORS.get(label,  f"C{labels.index(label)}")
            marker = MODEL_MARKERS.get(label, "o")
            ax2.plot(xs, ys, marker=marker, color=color, label=label,
                     linewidth=1.8, markersize=5)
            ax2.fill_between(xs,
                             [max(0, y - e) for y, e in zip(ys, errs)],
                             [min(1, y + e) for y, e in zip(ys, errs)],
                             color=color, alpha=0.12)

        ax2.set_xlabel("Attack Budget (fraction of edges)")
        ax2.set_title(ds.capitalize())
        ax2.set_xlim(-0.01, max(budgets_all) + 0.02)
        ax2.set_ylim(0, 1)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Test Accuracy under PGD Attack")
    handles = [
        mpatches.Patch(color=MODEL_COLORS.get(lbl, f"C{i}"), label=lbl)
        for i, (lbl, *_) in enumerate(model_configs)
    ]
    fig2.legend(handles=handles, loc="lower center", ncol=min(n_models, 5),
                fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig2.suptitle("Robustness under PGD Attack — RUNG vs RUNG_new vs Baselines",
                  fontsize=12, y=1.01)
    fig2.tight_layout()
    save_path2 = FIGURES_DIR / f"robustness_curves_{timestamp}.png"
    fig2.savefig(save_path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {save_path2.name}")

    # ── Figure 3: Robustness degradation (clean − attacked) ─────────────────
    # Shows how much each model "degrades" relative to its clean accuracy
    fig3, axes3 = plt.subplots(1, n_ds, figsize=(max(6, 4.5*n_ds), 4.5),
                               sharey=True)
    if n_ds == 1:
        axes3 = [axes3]

    for ax3, ds in zip(axes3, datasets):
        for label, model, norm, gamma in model_configs:
            # Clean baseline
            clean_d = _get_matrix("clean_acc", 0.0, ds)
            clean_entry = clean_d.get(label, (float("nan"), float("nan")))
            clean_val = clean_entry[0]
            if np.isnan(clean_val):
                continue

            degradations, stds_deg = [], []
            for bud in attack_budgets:
                d = _get_matrix("attacked_acc", bud, ds)
                entry = d.get(label, (float("nan"), float("nan")))
                if np.isnan(entry[0]):
                    degradations.append(None)
                    stds_deg.append(0.0)
                else:
                    degradations.append(clean_val - entry[0])
                    stds_deg.append(entry[1])

            valid_idx = [i for i, m in enumerate(degradations) if m is not None]
            if not valid_idx:
                continue
            xs    = [attack_budgets[i] for i in valid_idx]
            ys    = [degradations[i]   for i in valid_idx]
            errs  = [stds_deg[i]       for i in valid_idx]

            color  = MODEL_COLORS.get(label, f"C{labels.index(label)}")
            marker = MODEL_MARKERS.get(label, "o")
            ax3.plot(xs, ys, marker=marker, color=color, label=label,
                     linewidth=1.8, markersize=5)
            ax3.fill_between(xs,
                             [max(0, y - e) for y, e in zip(ys, errs)],
                             [y + e for y, e in zip(ys, errs)],
                             color=color, alpha=0.12)

        ax3.set_xlabel("Attack Budget")
        ax3.set_title(ds.capitalize())
        ax3.set_xlim(0, max(attack_budgets) + 0.02)
        ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)

    axes3[0].set_ylabel("Accuracy Degradation (clean − attacked)")
    fig3.legend(handles=handles, loc="lower center", ncol=min(n_models, 5),
                fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig3.suptitle("Accuracy Degradation under Attack", fontsize=12, y=1.01)
    fig3.tight_layout()
    save_path3 = FIGURES_DIR / f"degradation_{timestamp}.png"
    fig3.savefig(save_path3, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {save_path3.name}")

    # ── Figure 4: Summary heatmap — (model × dataset), colour = clean acc ───
    fig4, ax4 = plt.subplots(figsize=(max(5, n_ds * 1.4 + 1),
                                      max(3, n_models * 0.6 + 1)))
    matrix = np.full((n_models, n_ds), float("nan"))
    for i, (label, *_) in enumerate(model_configs):
        for j, ds in enumerate(datasets):
            d = _get_matrix("clean_acc", 0.0, ds)
            entry = d.get(label, (float("nan"), float("nan")))
            matrix[i, j] = entry[0]

    im = ax4.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlGn")
    ax4.set_xticks(range(n_ds));     ax4.set_xticklabels(datasets, fontsize=10)
    ax4.set_yticks(range(n_models)); ax4.set_yticklabels(labels, fontsize=10)
    for i in range(n_models):
        for j in range(n_ds):
            v = matrix[i, j]
            text = f"{v:.3f}" if not np.isnan(v) else "N/A"
            ax4.text(j, i, text, ha="center", va="center",
                     fontsize=9, color="black" if v > 0.4 else "white")
    plt.colorbar(im, ax=ax4, shrink=0.8, label="Clean Accuracy")
    ax4.set_title("Clean Accuracy Heatmap — all models × datasets")
    fig4.tight_layout()
    save_path4 = FIGURES_DIR / f"clean_heatmap_{timestamp}.png"
    fig4.savefig(save_path4, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: {save_path4.name}")

    print(f"\n  All figures saved in: {FIGURES_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare RUNG vs RUNG_new vs baselines and generate figures."
    )
    p.add_argument("--datasets", nargs="+", default=["cora", "citeseer"],
                   metavar="DATASET",
                   help="Datasets to evaluate (default: cora citeseer).")
    p.add_argument("--models", nargs="+", default=None,
                   metavar="LABEL",
                   help="Subset of model labels to run. "
                        "Choices: RUNG RUNG_new RUNG_new_SCAD GCN MLP. "
                        "Default: all.")
    p.add_argument("--budgets", nargs="+", type=float, default=ATTACK_BUDGETS,
                   metavar="B",
                   help="Attack budget fractions (default: 0.05 0.1 0.2 0.3 0.4).")
    p.add_argument("--max_epoch", type=int, default=300,
                   help="Training epochs (default: 300).")
    p.add_argument("--skip_clean",  action="store_true",
                   help="Skip training (use existing saved models).")
    p.add_argument("--skip_attack", action="store_true",
                   help="Skip attack evaluation.")
    p.add_argument("--skip_plot",   action="store_true",
                   help="Skip figure generation.")
    p.add_argument("--plot_only",   action="store_true",
                   help="Only generate figures from the latest CSV.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Filter model configs if --models specified
    configs = DEFAULT_MODEL_CONFIGS
    if args.models:
        allowed = set(args.models)
        configs = [c for c in configs if c[0] in allowed]
        if not configs:
            print(f"ERROR: none of {args.models} matched known labels "
                  f"{[c[0] for c in DEFAULT_MODEL_CONFIGS]}")
            sys.exit(1)

    if args.plot_only:
        # Load latest CSV and re-plot only
        latest = RESULTS_DIR / "comparison_latest.csv"
        if not latest.exists():
            print(f"ERROR: {latest} not found — run without --plot_only first.")
            sys.exit(1)
        import csv as _csv
        rows = []
        with open(latest) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                row["budget"]       = float(row["budget"])
                row["clean_acc"]    = float(row["clean_acc"])
                row["clean_std"]    = float(row["clean_std"])
                row["attacked_acc"] = float(row["attacked_acc"])
                row["attacked_std"] = float(row["attacked_std"])
                rows.append(row)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        datasets_in_csv = sorted({r["dataset"] for r in rows})
        _generate_plots(rows, datasets_in_csv, configs,
                        args.budgets, timestamp)
    else:
        run_comparison(
            datasets=args.datasets,
            model_configs=configs,
            attack_budgets=args.budgets,
            skip_clean=args.skip_clean,
            skip_attack=args.skip_attack,
            max_epoch=args.max_epoch,
            skip_plot=args.skip_plot,
        )
