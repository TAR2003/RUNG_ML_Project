#!/usr/bin/env python
"""
run_all.py — Train and/or attack one or more RUNG model strategies, writing logs.

Usage
-----
    # Train + attack RUNG (MCP, gamma=6.0) on cora and citeseer:
    python run_all.py --datasets cora citeseer --models RUNG

    # Run multiple strategies side-by-side (compound names accepted):
    python run_all.py --datasets cora citeseer --models RUNG RUNG_new_SCAD GCN MLP

    # Train only (skip PGD attack):
    python run_all.py --datasets cora citeseer --models RUNG --skip_attack

    # Attack only (requires models already saved by a previous clean run):
    python run_all.py --datasets cora citeseer --models RUNG --skip_clean

    # Use a custom norm / gamma for non-compound model names:
    python run_all.py --datasets cora --models RUNG --norm MCP --gamma 6.0

    # Longer training:
    python run_all.py --datasets cora --models RUNG --max_epoch 500

Compound model names (encode both architecture and norm in one token):
    RUNG_new_SCAD   RUNG_new_L1   RUNG_new_L2   RUNG_new_ADAPTIVE
    RUNG_SCAD       RUNG_L1       RUNG_L2

Standalone model names (pass as-is, no compound expansion):
    RUNG_learnable_gamma  — RUNG with per-layer learnable SCAD gamma parameters.
    RUNG_confidence_lambda — RUNG with learnable gamma AND per-node confidence lambda.

Logs are written to:
    log/<dataset>/clean/<model>_<norm>_<gamma>.log
    log/<dataset>/attack/<model>_norm<norm>_gamma<gamma>.log

To visualise all results from the logs run:
    python plot_logs.py
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Train and/or attack one or more model strategies across datasets."
)
parser.add_argument(
    "--datasets", nargs="+", default=["cora", "citeseer"],
    metavar="DATASET",
    help="Datasets to process (default: cora citeseer).",
)
parser.add_argument(
    "--models", nargs="+", default=["RUNG"],
    metavar="MODEL",
    help=(
        "Model strategies to run. Accepts compound names such as RUNG_new_SCAD. "
        "Default: RUNG."
    ),
)
parser.add_argument(
    "--norm", type=str, default="MCP",
    help="Norm / penalty used when the model name is not a compound name (default: MCP).",
)
parser.add_argument(
    "--gamma", type=float, default=6.0,
    help="Gamma hyperparameter (default: 6.0).",
)
parser.add_argument(
    "--max_epoch", type=int, default=300,
    help="Max training epochs (default: 300).",
)
parser.add_argument(
    "--gamma_init_strategy", type=str, default="uniform",
    choices=["uniform", "decreasing", "increasing"],
    help="Gamma initialisation strategy for RUNG_learnable_gamma (default: uniform).",
)
parser.add_argument(
    "--gamma_lr_factor", type=float, default=0.2,
    help="LR multiplier for gamma params in RUNG_learnable_gamma (default: 0.2).",
)
parser.add_argument(
    "--gamma_reg_strength", type=float, default=0.0,
    help="Gamma regularisation strength for RUNG_learnable_gamma (default: 0 = off).",
)
parser.add_argument(
    "--confidence_mode", type=str, default="protect_uncertain",
    choices=["protect_uncertain", "protect_confident", "symmetric"],
    help=(
        "Confidence-to-lambda mapping mode for RUNG_confidence_lambda. "
        "'protect_uncertain': uncertain nodes get higher lambda. "
        "'protect_confident': confident nodes get higher lambda. "
        "'symmetric': mid-confidence nodes get highest lambda."
    ),
)
parser.add_argument(
    "--alpha_init", type=float, default=1.0,
    help="Initial alpha sharpness for RUNG_confidence_lambda (default: 1.0).",
)
parser.add_argument(
    "--normalize_lambda", type=lambda x: x.lower() != 'false', default=True,
    help="Normalise per-node lambdas (RUNG_confidence_lambda, default: True).",
)
parser.add_argument(
    "--alpha_lr_factor", type=float, default=0.1,
    help="LR multiplier for alpha in RUNG_confidence_lambda (default: 0.1).",
)
parser.add_argument(
    "--alpha_reg_strength", type=float, default=0.001,
    help="Alpha regularisation strength for RUNG_confidence_lambda (default: 0.001).",
)
parser.add_argument(
    "--warmup_epochs", type=int, default=50,
    help="Warmup epochs (MLP-only) for RUNG_confidence_lambda (default: 50).",
)
parser.add_argument(
    "--skip_clean", action="store_true",
    help="Skip the clean-training step (use previously saved models).",
)
parser.add_argument(
    "--skip_attack", action="store_true",
    help="Skip the PGD-attack evaluation step.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PYTHON       = sys.executable
PROJECT_ROOT = Path(__file__).parent.resolve()


def _header(title: str) -> None:
    bar = "=" * (len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def _run(script: str, dataset: str, model: str) -> tuple[bool, float]:
    """
    Invoke `script` (clean.py or attack.py) for one (dataset, model) pair.
    `model` may be a compound name (e.g. RUNG_new_SCAD); the target script
    resolves it internally via its own _COMPOUND_MODEL_MAP.
    Returns (success, elapsed_seconds).
    """
    cmd = [
        PYTHON, str(PROJECT_ROOT / script),
        f"--model={model}",
        f"--norm={args.norm}",      # ignored by the script when model is compound
        f"--gamma={args.gamma}",
        f"--data={dataset}",
    ]
    if script == "clean.py":
        cmd.append(f"--max_epoch={args.max_epoch}")
        # Forward RUNG_learnable_gamma-specific args only when relevant.
        # clean.py ignores unknown args gracefully via argparse defaults.
        if model == "RUNG_learnable_gamma":
            cmd += [
                f"--gamma_init_strategy={args.gamma_init_strategy}",
                f"--gamma_lr_factor={args.gamma_lr_factor}",
                f"--gamma_reg_strength={args.gamma_reg_strength}",
            ]
        elif model == "RUNG_confidence_lambda":
            cmd += [
                f"--gamma_init_strategy={args.gamma_init_strategy}",
                f"--gamma_lr_factor={args.gamma_lr_factor}",
                f"--gamma_reg_strength={args.gamma_reg_strength}",
                f"--confidence_mode={args.confidence_mode}",
                f"--alpha_init={args.alpha_init}",
                f"--normalize_lambda={args.normalize_lambda}",
                f"--alpha_lr_factor={args.alpha_lr_factor}",
                f"--alpha_reg_strength={args.alpha_reg_strength}",
                f"--warmup_epochs={args.warmup_epochs}",
            ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - t0
    return proc.returncode == 0, elapsed


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
phases = []
if not args.skip_clean:
    phases.append(("clean.py",  "Train (clean)"))
if not args.skip_attack:
    phases.append(("attack.py", "PGD attack"))

if not phases:
    print("Nothing to do — both --skip_clean and --skip_attack were set.")
    sys.exit(0)

results: list[dict] = []
total_start = time.perf_counter()

for script, phase_label in phases:
    _header(f"{phase_label}  —  {len(args.models)} model(s) × {len(args.datasets)} dataset(s)")
    for model in args.models:
        for dataset in args.datasets:
            print(
                f">>> [{phase_label}]  dataset={dataset}  model={model}  "
                f"norm={args.norm}  gamma={args.gamma}",
                flush=True,
            )
            ok, elapsed = _run(script, dataset, model)
            tag = "DONE" if ok else "FAILED"
            print(f"<<< {tag}  ({elapsed:.1f}s)\n", flush=True)
            results.append(
                {"phase": phase_label, "model": model, "dataset": dataset,
                 "ok": ok, "elapsed": elapsed}
            )

total_elapsed = time.perf_counter() - total_start

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
_header("Summary")
print(f"  {'Phase':<16}  {'Model':<18}  {'Dataset':<12}  {'Status':<8}  {'Time':>8}")
print(f"  {'-'*16}  {'-'*18}  {'-'*12}  {'-'*8}  {'-'*8}")
for r in results:
    mark = "OK" if r["ok"] else "FAIL"
    print(
        f"  {r['phase']:<16}  {r['model']:<18}  {r['dataset']:<12}  "
        f"{mark:<8}  {r['elapsed']:>7.1f}s"
    )

n_ok   = sum(r["ok"] for r in results)
n_fail = len(results) - n_ok
print(f"\n  Total: {n_ok}/{len(results)} passed  |  wall-time: {total_elapsed:.1f}s")
if n_fail:
    print(
        f"\n  WARNING: {n_fail} job(s) failed. "
        "Check log/<dataset>/clean/ and log/<dataset>/attack/ for details.\n"
        "  Run  python plot_logs.py  to visualise all available results."
    )
else:
    print("\n  All jobs passed.  Run  python plot_logs.py  to visualise results.")

sys.exit(0 if n_fail == 0 else 1)
