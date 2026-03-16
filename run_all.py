#!/usr/bin/env python
"""
run_all.py — Train and/or attack one or more RUNG model strategies, writing logs.

MAIN USAGE (4-model comparison):
    python run_all.py --datasets cora citeseer --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined

    This trains + attacks all 4 models across Cora and Citeseer with extended budgets:
    [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

ADDITIONAL USAGE:
    # Train + attack RUNG (MCP, gamma=6.0) on cora and citeseer:
    python run_all.py --datasets cora citeseer --models RUNG

    # Run multiple strategies side-by-side:
    python run_all.py --datasets cora citeseer --models RUNG RUNG_new_SCAD

    # Train only (skip PGD attack):
    python run_all.py --datasets cora citeseer --models RUNG --skip_attack

    # Attack only (requires models already saved by a previous clean run):
    python run_all.py --datasets cora citeseer --models RUNG --skip_clean

    # Use a custom norm / gamma for non-compound model names:
    python run_all.py --datasets cora --models RUNG --norm MCP --gamma 6.0

    # Longer training:
    python run_all.py --datasets cora --models RUNG --max_epoch 500

SUPPORTED MODELS:
    Base:
        RUNG, RUNG_new, RUNG_new_SCAD, RUNG_new_L1, RUNG_new_L2, RUNG_new_ADAPTIVE
        RUNG_SCAD, RUNG_L1, RUNG_L2

    Advanced variants (4-model comparison):
        RUNG_percentile_gamma     — Percentile-based adaptive gamma per layer
        RUNG_learnable_distance   — Learnable distance metric (cosine/projection/bilinear)
        RUNG_combined             — Percentile gamma + cosine distance

    Other:
        RUNG_learnable_gamma, RUNG_parametric_gamma, RUNG_confidence_lambda
        GCN, GAT, APPNP, L1, MLP

LOGS:
    log/<dataset>/clean/<model>_<norm>_<gamma>.log
    log/<dataset>/attack/<model>_norm<norm>_gamma<gamma>.log

VISUALISE:
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
        "Model strategies to run. Examples: RUNG, RUNG_percentile_gamma, "
        "RUNG_learnable_distance, RUNG_combined, RUNG_new_SCAD, etc. "
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

# ===== EXTENDED ATTACK BUDGETS =====
parser.add_argument(
    "--budgets", type=float, nargs="+", 
    default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    metavar="BUDGET",
    help=(
        "Attack budgets (as fraction of edges). "
        "Default: 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00"
    ),
)

# ===== RUNG_learnable_gamma parameters =====
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

# ===== RUNG_parametric_gamma parameters =====
parser.add_argument(
    "--decay_rate_init", type=float, default=0.85,
    help="Initial decay rate per layer for RUNG_parametric_gamma (default: 0.85).",
)
parser.add_argument(
    "--decay_rate_reg_strength", type=float, default=0.0,
    help="Decay rate regularisation for RUNG_parametric_gamma (default: 0.0).",
)

# ===== RUNG_confidence_lambda parameters =====
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

# ===== RUNG_percentile_gamma parameters (same used for RUNG_combined & RUNG_learnable_distance) =====
parser.add_argument(
    "--percentile_q",
    type=float,
    default=0.75,
    help=(
        "Percentile for gamma computation in RUNG_percentile_gamma, "
        "RUNG_learnable_distance, and RUNG_combined. "
        "gamma^(k) = quantile(y_edges^(k), percentile_q). "
        "Range (0, 1). Higher = lighter pruning. "
        "Default: 0.75."
    ),
)
parser.add_argument(
    "--use_layerwise_q",
    type=lambda x: x.lower() != 'false',
    default=False,
    help=(
        "If True, use different percentile_q for early vs late layers. "
        "Early layers (first half) use --percentile_q. "
        "Late layers (second half) use --percentile_q_late. "
        "Default: False."
    ),
)
parser.add_argument(
    "--percentile_q_late",
    type=float,
    default=0.65,
    help=(
        "Percentile q for late layers when use_layerwise_q=True. "
        "Default: 0.65."
    ),
)

# ===== RUNG_learnable_distance parameters =====
parser.add_argument(
    "--distance_mode", type=str, default="cosine",
    choices=["cosine", "projection", "bilinear"],
    help=(
        "Distance metric for RUNG_learnable_distance. "
        "'cosine': scale-invariant (default, recommended). "
        "'projection': learnable MLP projection. "
        "'bilinear': learnable linear projection."
    ),
)
parser.add_argument(
    "--proj_dim", type=int, default=32,
    help="Projection dimension for projection/bilinear modes (default: 32).",
)
parser.add_argument(
    "--dist_lr_factor", type=float, default=0.5,
    help="LR multiplier for distance module in RUNG_learnable_distance (default: 0.5).",
)

# ===== Control flags =====
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
    Invoke appropriate script for one (dataset, model) pair.
    
    Routing:
    - RUNG_combined: uses train_test_combined.py (combined clean + attack)
    - Others: uses clean.py + attack.py as separate phases
    
    Returns (success, elapsed_seconds).
    """
    
    # Special handling for RUNG_combined which has its own integrated script
    if model == "RUNG_combined" and script == "clean.py":
        # Use train_test_combined.py for RUNG_combined instead
        # Pass budgets to train_test_combined.py (centrally controlled from run_all.py)
        cmd = [
            PYTHON, str(PROJECT_ROOT / "train_test_combined.py"),
            f"--dataset={dataset}",
            f"--percentile_q={args.percentile_q}",
            f"--use_layerwise_q={args.use_layerwise_q}",
            f"--percentile_q_late={args.percentile_q_late}",
            f"--max_epoch={args.max_epoch}",
            "--budgets",
        ] + [str(b) for b in args.budgets]
        
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = time.perf_counter() - t0
        return proc.returncode == 0, elapsed
    
    elif model == "RUNG_combined" and script == "attack.py":
        # Skip attack step for RUNG_combined (already done in train_test_combined.py)
        return True, 0.0
    
    # Standard handling for other models: clean.py and attack.py
    cmd = [
        PYTHON, str(PROJECT_ROOT / script),
        f"--model={model}",
        f"--norm={args.norm}",      # ignored if model is compound
        f"--gamma={args.gamma}",
        f"--data={dataset}",
    ]
    
    if script == "clean.py":
        cmd.append(f"--max_epoch={args.max_epoch}")
        
        # Model-specific parameters
        if model == "RUNG_learnable_gamma":
            cmd += [
                f"--gamma_init_strategy={args.gamma_init_strategy}",
                f"--gamma_lr_factor={args.gamma_lr_factor}",
                f"--gamma_reg_strength={args.gamma_reg_strength}",
            ]
        
        elif model == "RUNG_parametric_gamma":
            cmd += [
                f"--decay_rate_init={args.decay_rate_init}",
                f"--gamma_lr_factor={args.gamma_lr_factor}",
                f"--decay_rate_reg_strength={args.decay_rate_reg_strength}",
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
        
        elif model == "RUNG_percentile_gamma":
            cmd += [
                f"--percentile_q={args.percentile_q}",
                f"--use_layerwise_q={args.use_layerwise_q}",
                f"--percentile_q_late={args.percentile_q_late}",
            ]
        
        elif model == "RUNG_learnable_distance":
            cmd += [
                f"--percentile_q={args.percentile_q}",
                f"--use_layerwise_q={args.use_layerwise_q}",
                f"--percentile_q_late={args.percentile_q_late}",
                f"--distance_mode={args.distance_mode}",
                f"--proj_dim={args.proj_dim}",
                f"--dist_lr_factor={args.dist_lr_factor}",
            ]
    
    elif script == "attack.py":
        # Pass the extended budget array to attack.py
        cmd.extend(["--budgets"] + [str(b) for b in args.budgets])
    
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

# Separate RUNG_combined from other models
rung_combined_models = [m for m in args.models if m == "RUNG_combined"]
other_models = [m for m in args.models if m != "RUNG_combined"]

# Process other models with standard clean.py + attack.py pipeline
if other_models:
    for script, phase_label in phases:
        _header(
            f"{phase_label}  —  {len(other_models)} model(s) × {len(args.datasets)} dataset(s)"
        )
        for model in other_models:
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
                    {
                        "phase": phase_label, "model": model, "dataset": dataset,
                        "ok": ok, "elapsed": elapsed, "script": script
                    }
                )

# Process RUNG_combined with integrated train_test_combined.py pipeline
if rung_combined_models and not args.skip_clean:
    _header(
        f"Train + Attack (unified)  —  {len(rung_combined_models)} model(s) × {len(args.datasets)} dataset(s)"
    )
    for model in rung_combined_models:
        for dataset in args.datasets:
            print(
                f">>> [Train + Attack]  dataset={dataset}  model={model}  "
                f"percentile_q={args.percentile_q}",
                flush=True,
            )
            ok, elapsed = _run("clean.py", dataset, model)  # Calls train_test_combined.py internally
            tag = "DONE" if ok else "FAILED"
            print(f"<<< {tag}  ({elapsed:.1f}s)\n", flush=True)
            results.append(
                {
                    "phase": "Train + Attack (unified)",
                    "model": model, "dataset": dataset,
                    "ok": ok, "elapsed": elapsed, "script": "train_test_combined.py"
                }
            )

total_elapsed = time.perf_counter() - total_start

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
_header("Summary")
print(
    f"  {'Phase':<30}  {'Model':<20}  {'Dataset':<12}  "
    f"{'Script':<25}  {'Status':<8}  {'Time':>8}"
)
print(
    f"  {'-'*30}  {'-'*20}  {'-'*12}  "
    f"{'-'*25}  {'-'*8}  {'-'*8}"
)
for r in results:
    mark = "OK" if r["ok"] else "FAIL"
    script = r.get("script", "?")
    print(
        f"  {r['phase']:<30}  {r['model']:<20}  {r['dataset']:<12}  "
        f"{script:<25}  {mark:<8}  {r['elapsed']:>7.1f}s"
    )

n_ok   = sum(r["ok"] for r in results)
n_fail = len(results) - n_ok
print(
    f"\n  Total: {n_ok}/{len(results)} passed  |  wall-time: {total_elapsed:.1f}s\n"
)

# Model-specific notes
print(f"  Model-specific configuration:")
print(f"    RUNG & RUNG_learnable_gamma:  MCP/SCAD penalty, gamma={args.gamma}")
print(f"    RUNG_percentile_gamma:        Percentile-based gamma, percentile_q={args.percentile_q}")
print(f"    RUNG_learnable_distance:      Distance={args.distance_mode}, percentile_q={args.percentile_q}")
print(f"    RUNG_combined:                Cosine distance + percentile_q={args.percentile_q}")
print(f"    Attack budgets:               {args.budgets}")
print(f"\n")

if n_fail:
    print(
        f"  WARNING: {n_fail} job(s) failed. "
        "Check log/<dataset>/clean/ and log/<dataset>/attack/ for details.\n"
        "  Run  python plot_logs.py  to visualise all available results."
    )
else:
    print(
        "  ✅ All jobs passed!"
        "\n  Run  python plot_logs.py  to visualise and compare all 4 models."
    )

sys.exit(0 if n_fail == 0 else 1)
