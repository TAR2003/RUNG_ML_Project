#!/usr/bin/env python
"""
run_all.py — Train, evaluate, and compare RUNG models in one shot.

Usage
-----
    # Full run (train + attack, all datasets, default RUNG/MCP/gamma=6.0):
    python run_all.py

    # Compare RUNG (original) vs RUNG_new (refactored) vs GCN/MLP baselines:
    python run_all.py --compare

    # Compare on a specific subset of datasets:
    python run_all.py --compare --datasets cora citeseer

    # Compare specific models only:
    python run_all.py --compare --compare_models RUNG RUNG_new GCN

    # Skip training in compare mode (use existing saved models):
    python run_all.py --compare --skip_clean

    # Only regenerate figures from the last comparison CSV:
    python run_all.py --compare --plot_only

    # Train only (skip PGD attack):
    python run_all.py --skip_attack

    # Attack only (requires models already saved by clean.py):
    python run_all.py --skip_clean

    # Run a specific subset of datasets:
    python run_all.py --datasets cora citeseer

    # Run with a different norm/gamma:
    python run_all.py --norm SCAD --gamma 6.0

    # Use a specific model:
    python run_all.py --model RUNG --norm MCP --gamma 6.0

Results in compare mode
-----------------------
    results/comparison/comparison_latest.csv   — aggregated CSV
    figures/comparison/clean_accuracy_*.png    — bar chart
    figures/comparison/robustness_curves_*.png — accuracy vs budget
    figures/comparison/degradation_*.png       — accuracy drop vs budget
    figures/comparison/clean_heatmap_*.png     — heatmap summary
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset lists
# ---------------------------------------------------------------------------
HOMOPHILIC_DATASETS   = ["cora", "citeseer"]
HETEROPHILIC_DATASETS = ["chameleon", "squirrel", "actor", "cornell", "texas", "wisconsin"]
ALL_DATASETS          = HOMOPHILIC_DATASETS + HETEROPHILIC_DATASETS

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run clean training and/or PGD-attack evaluation on all datasets."
)
parser.add_argument("--datasets",    nargs="+", default=ALL_DATASETS,
                    metavar="DATASET",
                    help="Datasets to process. Default: all (cora citeseer chameleon "
                         "squirrel actor cornell texas wisconsin).")
parser.add_argument("--model",       type=str, default="RUNG",
                    help="Model name (default: RUNG).")
parser.add_argument("--norm",        type=str, default="MCP",
                    help="Norm / penalty (default: MCP).")
parser.add_argument("--gamma",       type=float, default=6.0,
                    help="Gamma hyperparameter (default: 6.0).")
parser.add_argument("--max_epoch",   type=int, default=300,
                    help="Max training epochs (default: 300).")
parser.add_argument("--skip_clean",  action="store_true",
                    help="Skip the clean-training step.")
parser.add_argument("--skip_attack", action="store_true",
                    help="Skip the PGD-attack evaluation step.")

# ── Comparison / multi-model flags ──────────────────────────────────────────
parser.add_argument("--compare", action="store_true",
                    help="Run multi-model comparison: RUNG (original) vs "
                         "RUNG_new (refactored) vs GCN/MLP baselines. "
                         "Results and figures saved under results/comparison/ and "
                         "figures/comparison/. All other flags still apply.")
parser.add_argument("--compare_models", nargs="+", default=None,
                    metavar="LABEL",
                    help="Which models to include in the comparison. "
                         "Choices: RUNG RUNG_new RUNG_new_SCAD GCN MLP. "
                         "Default: all five.")
parser.add_argument("--skip_plot", action="store_true",
                    help="(compare mode) Skip figure generation after running.")
parser.add_argument("--plot_only", action="store_true",
                    help="(compare mode) Only regenerate figures from the last "
                         "saved CSV; skip training and attack.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PYTHON = sys.executable   # same interpreter that launched this script
PROJECT_ROOT = Path(__file__).parent.resolve()


def _run(script: str, dataset: str, extra: list[str] | None = None) -> tuple[bool, float]:
    """Run `script` with standard args for `dataset`. Returns (success, elapsed_seconds)."""
    cmd = [
        PYTHON, str(PROJECT_ROOT / script),
        f"--model={args.model}",
        f"--norm={args.norm}",
        f"--gamma={args.gamma}",
        f"--data={dataset}",
    ]
    if script == "clean.py":
        cmd.append(f"--max_epoch={args.max_epoch}")
    if extra:
        cmd.extend(extra)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0
    return result.returncode == 0, elapsed


def _header(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}\n", flush=True)


def _row(label: str, status: bool, elapsed: float) -> str:
    mark = "✓" if status else "✗"
    return f"  {mark}  {label:<35}  {elapsed:>7.1f}s"


# ---------------------------------------------------------------------------
# ── COMPARE MODE — RUNG vs RUNG_new vs baselines ────────────────────────────
# ---------------------------------------------------------------------------
if args.compare or args.plot_only:
    # Default datasets for comparison: homophilic only (fast; heterophilic needs
    # torch_geometric which may not always be available)
    compare_datasets = args.datasets
    if compare_datasets == ALL_DATASETS:
        compare_datasets = HOMOPHILIC_DATASETS   # default to cora + citeseer

    _header("COMPARE MODE — RUNG (original) vs RUNG_new vs baselines")
    print(f"  Datasets : {compare_datasets}")
    print(f"  Models   : {args.compare_models or 'all defaults'}")
    print(f"  Results  → results/comparison/comparison_latest.csv")
    print(f"  Figures  → figures/comparison/\n")

    # Import and call the comparison runner directly (same Python process)
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.compare_models import run_comparison, DEFAULT_MODEL_CONFIGS, _generate_plots, RESULTS_DIR
    import csv as _csv
    from datetime import datetime

    # Filter model configs if --compare_models specified
    configs = DEFAULT_MODEL_CONFIGS
    if args.compare_models:
        allowed = set(args.compare_models)
        configs = [c for c in configs if c[0] in allowed]
        if not configs:
            print(f"ERROR: none of {args.compare_models} matched known labels "
                  f"{[c[0] for c in DEFAULT_MODEL_CONFIGS]}")
            sys.exit(1)

    if args.plot_only:
        latest = RESULTS_DIR / "comparison_latest.csv"
        if not latest.exists():
            print(f"ERROR: {latest} not found — run without --plot_only first.")
            sys.exit(1)
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
        _generate_plots(rows, compare_datasets, configs,
                        [0.05, 0.10, 0.20, 0.30, 0.40], timestamp)
        print("\nDone — figures regenerated from existing CSV.")
        sys.exit(0)

    csv_path = run_comparison(
        datasets=compare_datasets,
        model_configs=configs,
        attack_budgets=[0.05, 0.10, 0.20, 0.30, 0.40],
        skip_clean=args.skip_clean,
        skip_attack=args.skip_attack,
        max_epoch=args.max_epoch,
        skip_plot=args.skip_plot,
    )
    print(f"\nComparison complete.  CSV: {csv_path}")
    sys.exit(0)


# ---------------------------------------------------------------------------
# ── SINGLE-MODEL MODE (original run_all behaviour) ───────────────────────────
# ---------------------------------------------------------------------------
results: list[dict] = []   # {dataset, phase, ok, elapsed}

phases = []
if not args.skip_clean:
    phases.append(("clean.py",  "Train (clean)"))
if not args.skip_attack:
    phases.append(("attack.py", "PGD attack    "))

if not phases:
    print("Nothing to do (both --skip_clean and --skip_attack were set).")
    sys.exit(0)

total_start = time.perf_counter()

for script, phase_label in phases:
    _header(f"{phase_label}  —  {len(args.datasets)} dataset(s)")
    for dataset in args.datasets:
        print(f">>> [{phase_label.strip()}]  dataset={dataset}  "
              f"model={args.model}  norm={args.norm}  gamma={args.gamma}")
        ok, elapsed = _run(script, dataset)
        status_str = "DONE" if ok else "FAILED"
        print(f"<<< {status_str}  ({elapsed:.1f}s)\n", flush=True)
        results.append({"dataset": dataset, "phase": phase_label.strip(),
                        "ok": ok, "elapsed": elapsed})

total_elapsed = time.perf_counter() - total_start

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
_header("Summary")
print(f"  {'Phase':<20}  {'Dataset':<12}  {'Status':<8}  {'Time':>8}")
print(f"  {'-'*20}  {'-'*12}  {'-'*8}  {'-'*8}")
for r in results:
    mark = "✓ OK" if r["ok"] else "✗ FAIL"
    print(f"  {r['phase']:<20}  {r['dataset']:<12}  {mark:<8}  {r['elapsed']:>7.1f}s")

n_ok   = sum(r["ok"] for r in results)
n_fail = len(results) - n_ok
print(f"\n  Total: {n_ok}/{len(results)} passed  |  wall-time: {total_elapsed:.1f}s")
if n_fail:
    print(f"\n  WARNING: {n_fail} job(s) failed. "
          "Check the log files in log/<dataset>/ for details.")

sys.exit(0 if n_fail == 0 else 1)
