# 006 — Visualization and Plotting Module

## Date
2026-02-27

## What Changed

### New Files
- **`experiments/plot_results.py`**: Full plotting module
- **`figures/`**: Output directory (created on first save)

## Figures Implemented

| Function                        | Reproduces         | New? | Input required              |
|---------------------------------|--------------------|------|-----------------------------|
| `plot_bias_simulation()`        | Figure 2           | No   | None (runs simulation)      |
| `plot_robustness_curves()`      | Figure 1 style     | YES  | penalty_comparison CSV      |
| `plot_bias_curves()`            | Figure 6           | No   | bias_curve CSV              |
| `plot_gamma_heatmap()`          | Figure 12          | No   | gamma_sensitivity CSV       |
| `plot_layer_sensitivity()`      | Figure 13          | No   | num_layers CSV              |
| `plot_homophily_vs_performance()` | —               | YES  | heterophilic CSV            |
| `plot_penalty_comparison_bars()`  | —               | YES  | penalty_comparison CSV      |

## Global Style
All figures use NeurIPS-consistent aesthetics:
- Serif font, 11pt base size
- No top/right spines (clean framing)
- Consistent color scheme:
  - Red   `#d62728` → L2 / GCN / APPNP
  - Blue  `#1f77b4` → L1 / biased baseline
  - Green `#2ca02c` → MCP / RUNG (default)
  - Orange `#ff7f0e` → SCAD
  - Purple `#9467bd` → Adaptive

## Figure Details

### `plot_bias_simulation()` (Figure 2)
Generates 100 clean samples from N((0,0), I) and outliers from N((8,8), 0.5I).
Runs three estimators:
- l2 mean (arithmetic mean)
- l1 mean (Weiszfeld algorithm for geometric median)
- MCP mean (IRLS with MCP penalty, γ=2.0)

### `plot_robustness_curves()` (NEW-1)
Reads from `penalty_comparison` experiment CSV.
Shows accuracy ± std vs attack budget for each penalty, one subplot per dataset.
Error bars computed across random seeds.

### `plot_bias_curves()` (Figure 6)
Reads from `bias_curve` experiment CSV (requires `measure_bias=True`).
Shows Σ_i ||f_i - f_i*||² vs attack budget with shaded ±1σ bands.

### `plot_gamma_heatmap()` (Figure 12)
Reads from `gamma_sensitivity` experiment CSV.
Shows 2D heatmap of accuracy over (γ × λ̂) at a fixed attack budget.

### `plot_layer_sensitivity()` (Figure 13)
Reads from `num_layers` experiment CSV.
Shows accuracy vs number of propagation layers for each penalty.

### `plot_homophily_vs_performance()` (NEW-2)
Reads from `heterophilic` experiment CSV.
Scatter plot of clean accuracy vs edge homophily ratio h across datasets.
Dataset names are annotated on each point.
Vertical dashed line at h=0.5 separates homophilic from heterophilic regime.

### `plot_penalty_comparison_bars()` (NEW-3)
Reads from any experiment CSV.
Bar chart comparing clean accuracy (budget=0) across all penalties on a single dataset.

## How to Use

```bash
# Generate Figure 2 (no CSV needed):
python experiments/plot_results.py --figure bias_simulation --save_dir figures/

# Penalty robustness curves (after running penalty_comparison):
python experiments/plot_results.py \
    --figure robustness \
    --results_csv results/penalty_comparison_20260227.csv \
    --save_dir figures/

# Bias curves (after running bias_curve with measure_bias=True):
python experiments/plot_results.py \
    --figure bias_curves \
    --results_csv results/bias_curve_20260227.csv

# Homophily analysis (after running heterophilic):
python experiments/plot_results.py \
    --figure homophily \
    --results_csv results/heterophilic_20260227.csv

# Gamma heatmap at budget=20% (after running gamma_sensitivity):
python experiments/plot_results.py \
    --figure gamma_heatmap \
    --results_csv results/gamma_sensitivity_20260227.csv \
    --budget 20
```

## Output Format
All figures are saved as PDF (vector graphics, suitable for papers).
The `--save_dir` argument controls the output directory.
