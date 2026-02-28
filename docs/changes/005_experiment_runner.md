# 005 — Systematic Experiment Runner

## Date
2026-02-27

## Motivation
Manual experiment execution leads to inconsistent configurations and lost results.
A single declarative experiment runner ensures reproducibility, handles failures
gracefully, and saves results incrementally to CSV.

## What Changed

### New Files
- **`experiments/run_ablation.py`**: Full ablation runner
- **`results/`**: Output directory (created by runner on first use)

## Experiments Defined

| Name               | Description                                        | Est. Runtime |
|--------------------|----------------------------------------------------|--------------|
| penalty_comparison | MCP vs SCAD vs L1 vs L2 on Cora/Citeseer           | ~2–4 hours   |
| heterophilic       | All penalties on heterophilic datasets             | ~4–8 hours   |
| bias_curve         | Replicate+extend Figure 6 (bias vs budget)         | ~1–2 hours   |
| gamma_sensitivity  | Replicate+extend Figure 12 (γ vs λ heatmap)        | ~2–3 hours   |
| num_layers         | Replicate Figure 13 (performance vs layers)        | ~1–2 hours   |

## Architecture

### `run_single_experiment(config)`
Wires together:
1. `get_dataset(dataset)` → (A, X, y)
2. `get_splits(y)` → 5 train/val/test splits
3. `get_model_default(...)` → (model, fit_params)
4. `fit(model, A, X, y, ...)` → trains for 300 epochs
5. `pgd_attack(model, A, X, y, ...)` → perturbs graph if budget > 0
6. Accuracy evaluation on perturbed graph
7. Optional: `compute_estimation_bias(...)` → bias metrics

### `run_ablation(experiment_name, results_dir)`
- Generates all (dataset × penalty × gamma × lam_hat × budget × seed × layers) combos
- Runs `run_single_experiment` for each
- Writes results incrementally to CSV (safe against crashes)
- Prints progress with accuracy, bias, and runtime per configuration

## Files Changed
- `experiments/run_ablation.py`: New file (full implementation)
- `results/`: New directory

## How to Use

```bash
# Run penalty comparison (recommended first experiment)
python experiments/run_ablation.py --experiment penalty_comparison

# Run heterophilic evaluation (requires torch_geometric)
python experiments/run_ablation.py --experiment heterophilic --results_dir ./results/hetero

# Replicate Figure 6 bias curves
python experiments/run_ablation.py --experiment bias_curve

# Replicate Figure 12 gamma sensitivity heatmap
python experiments/run_ablation.py --experiment gamma_sensitivity

# Replicate Figure 13 layer sensitivity
python experiments/run_ablation.py --experiment num_layers
```

## Output Format

CSV columns:
```
dataset, penalty, gamma, lambda_hat, budget, seed, num_layers,
accuracy, std, bias_total, bias_mean, runtime_seconds
```

- `accuracy`: mean accuracy over 5 splits (float)
- `std`: standard deviation over splits (float)
- `bias_total`: Σ_i ||f_i - f_i*||² (NaN if measure_bias=False)
- `budget`: global attack budget as % of total edges
- Failed configurations are written as NaN rows (not silently dropped)

## Penalty Name Convention

The runner uses uppercase penalty names matching `get_model_default`:
- `'MCP'`      → RUNG (paper default)
- `'SCAD'`     → SCAD variant
- `'L1'`       → RUNG-l1 (biased baseline)
- `'L2'`       → APPNP-equivalent
- `'ADAPTIVE'` → Homophily-aware (for heterophilic datasets)
