# 002 — Heterophilic Dataset Support

## Date
2026-02-27

## Motivation
The RUNG paper (Section 6) explicitly identifies heterophilic graphs as a
limitation and future work direction. On heterophilic graphs, connected nodes
tend to have DIFFERENT classes, which violates the assumption behind MCP
(that large feature differences = adversarial). Adding these datasets enables
us to test whether RUNG's unbiased aggregation still helps when the inductive
bias is wrong.

## Datasets Added

| Dataset   | Source          | Nodes  | Edges   | Classes | Homophily (approx) |
|-----------|-----------------|--------|---------|---------|---------------------|
| Chameleon | WikipediaNetwork| 2,277  | 36,101  | 5       | ~0.23               |
| Squirrel  | WikipediaNetwork| 5,201  | 217,073 | 5       | ~0.22               |
| Actor     | Actor (Film)    | 7,600  | 33,544  | 5       | ~0.22               |
| Cornell   | WebKB           | 183    | 295     | 5       | ~0.20               |
| Texas     | WebKB           | 183    | 309     | 5       | ~0.11               |
| Wisconsin | WebKB           | 251    | 499     | 5       | ~0.21               |

## What Changed

### Modified Files
- **`train_eval_data/get_dataset.py`**:
  - **Removed** stale stub functions (`get_datasplit`, `get_target_node_idx`)
  - **Added** `HETEROPHILIC_DATASETS` list (top of file)
  - **Added** `HOMOPHILY_RATIOS` dict (approximate values from literature)
  - **Extended** `get_dataset()` dispatcher: added `elif name in HETEROPHILIC_DATASETS`
    branch that calls `_load_or_download_heterophilic()` — lines ~35–40
  - **Added** `get_homophily_ratio(A, y)` — computes measured h from dense adj
  - **Added** `_load_or_download_heterophilic(name, root)` — downloads via
    torch_geometric on first call, caches as `.pt` files in `data/heter_data/{name}/`
  - **Added** `load_heterophilic_dataset(name, root, split_seed)` — public API
    returning `(A, X, y, homophily)`

## Cache Location
Downloaded datasets are cached at:
```
data/heter_data/{name}/adj.pt     — [N, N] dense float32 adjacency
data/heter_data/{name}/fea.pt     — [N, F] float32 features
data/heter_data/{name}/label.pt   — [N] int64 labels
```
This matches the existing `else` branch format in `get_dataset()`.

## Data Split Protocol
Uses the same `get_splits(y)` function as Cora/Citeseer:
5 deterministic stratified 10-10-80 splits (no change to split logic).
Heterophilic datasets may have class-imbalance issues with stratification;
a warning is printed if stratification fails.

## Dependency
Heterophilic datasets require `torch_geometric`. Install with:
```bash
pip install torch_geometric
```
If not installed, an informative `ImportError` is raised on first access.

## How to Use
```bash
# Download happens automatically on first run:
python clean.py --model='RUNG' --penalty=mcp --gamma=3.0 --data='chameleon'

# Explicitly download and inspect:
python -c "
from train_eval_data.get_dataset import load_heterophilic_dataset
A, X, y, h = load_heterophilic_dataset('chameleon')
print(f'Homophily: {h:.4f}')
"
```

## Expected Observations
- RUNG with MCP should degrade on heterophilic datasets (h < 0.3) because
  MCP penalizes large feature differences, which are LEGITIMATE for cross-class edges.
- The adaptive penalty (Task 5 / `--penalty=adaptive`) should recover performance
  by using inverted weights for predicted-heterophilic edges.
- For Texas (h=0.11), performance degradation is expected to be most severe.
