# 003 — Estimation Bias Measurement

## Date
2026-02-27

## Motivation
Section 4.3 and Figure 6 of the RUNG paper show that l1-based GNNs accumulate
estimation bias as attack budget grows, while RUNG-MCP maintains near-zero bias.
This metric is the core empirical evidence for RUNG's advantage. Implementing
it as a reusable module enables systematic comparison across all penalty types
and heterophilic datasets.

## Mathematical Definition

**Estimation Bias:**
$$\text{Bias} = \sum_{i \in V} \|f_i - f_i^*\|_2^2$$

where:
- $f_i^*$ = aggregated node feature on the **clean** graph
- $f_i$   = aggregated node feature on the **attacked** graph

**Key insight:** In the RUNG architecture, propagation is entirely in class space —
the MLP maps X → F⁰ (initial logits), then QN-IRLS propagates F⁰. Therefore
`get_aggregated_features(A, X)` returns the final logit matrix, which IS the
aggregated feature. No separate classification head exists.

## What Changed

### New Files
- **`utils/metrics.py`** (new): Estimation bias and distribution metrics
  - `compute_estimation_bias(model, A_clean, X, A_attacked, device)` — lines 50–96
    Returns `(bias_total, bias_mean)` or `(bias_total, bias_mean, bias_per_node)`
  - `compute_bias_curve(model, A_clean, X, attacked_graphs_by_budget, device)` — lines 99–140
    Replicates Figure 6; returns dict `{budget → {'bias_total': ..., 'bias_mean': ...}}`
  - `compute_edge_feature_diff_distribution(model, A, X, device, num_bins)` — lines 143–196
    Replicates Figure 7; returns `(hist_values, bin_edges, mean_diff)`
  - `compute_robust_accuracy(model, A_attacked, X, y, test_idx, device)` — lines 199–228
  - `compute_clean_and_attacked_accuracy(...)` — lines 231–244

- **`utils/__init__.py`** (new package init): Re-exports all symbols from
  `utils.py` (to avoid shadowing) AND imports from `utils/metrics.py`.
  Existing `from utils import add_loops, ...` imports continue to work.

### Modified Files
- **`model/rung.py`**: Added `get_aggregated_features(self, A, X)` method
  (returns `self.forward(A, X)`, since logits ARE the aggregated features).

## Files Changed
- `utils/metrics.py`: New file (full implementation)
- `utils/__init__.py`: New package init (re-exports utils.py + metrics)
- `model/rung.py`: Added `get_aggregated_features()` method

## How to Use

```python
import torch
from utils.metrics import compute_bias_curve

# Assuming you have:
#   model    — trained RUNG model
#   A_clean  — [N, N] clean adjacency
#   X        — [N, D] features
#   attacked_budgets — dict {0.05: A_5pct, 0.10: A_10pct, ...}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

bias_results = compute_bias_curve(
    model=model,
    A_clean=A_clean,
    X=X,
    attacked_graphs_by_budget=attacked_budgets,
    device=device,
)
# bias_results[0.10]['bias_total'] → float
# bias_results[0.10]['bias_mean']  → float
```

## Expected Behavior (replicating Figure 6)
- **l1 penalty**: bias increases linearly with attack budget
- **MCP penalty (RUNG)**: bias stays near zero across all budgets
- **SCAD penalty**: should behave similarly to MCP (hypothesis to verify)
- **adaptive penalty on heterophilic**: bias expected to be lower than MCP
