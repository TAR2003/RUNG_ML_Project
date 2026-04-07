# Changes from Original RUNG to Handle Heterophilic Graphs

This document explains, in implementation terms, what was changed from original RUNG to better handle heterophilic graphs in this codebase.

## Files Used as Ground Truth

- `model/rung.py`
- `model/penalty.py`
- `model/rung_homophily_adaptive.py`
- `exp/config/get_model.py`
- `clean.py`
- `train_eval_data/fit_homophily_adaptive.py`
- `train_eval_data/get_dataset.py`

---

## 1) Why Original RUNG Needs Adaptation for Heterophily

Original RUNG (default MCP behavior) uses feature dissimilarity as an anomaly signal. In homophilic graphs this is usually fine: cross-class or corrupted edges tend to look dissimilar.

In heterophilic graphs, however, many valid edges are naturally cross-class and can have large feature difference. So treating large distance as suspicious can over-prune legitimate edges.

The original forward path in `model/rung.py` is:

```python
# model/rung.py
Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
y = Z.sqrt()

if self.penalty == 'adaptive':
    soft_labels = torch.softmax(F.detach(), dim=-1)
    W = PenaltyFunction.homophily_adaptive(y, self.gamma, soft_labels)
else:
    W = self.w(y)
```

The codebase introduces two practical heterophily-oriented changes on top of this baseline idea.

---

## 2) First Heterophily Change: Adaptive Penalty in Base RUNG (`penalty='adaptive'`)

## 2.1 What Changed

The first change keeps the original `RUNG` class, but introduces a heterophily-aware branch in `model/rung.py` by selecting `penalty='adaptive'`.

```python
# model/rung.py
if self.penalty == 'adaptive':
    soft_labels = torch.softmax(F.detach(), dim=-1)
    W = PenaltyFunction.homophily_adaptive(y, self.gamma, soft_labels)
```

This redirects weighting to `PenaltyFunction.homophily_adaptive` in `model/penalty.py`.

## 2.2 Exact Heterophily-Adaptive Weight Formula in Code

```python
# model/penalty.py
dot_product = soft_labels @ soft_labels.t()
h = 1.0 - dot_product.clamp(0.0, 1.0)

y_safe = y.clamp(min=eps)
W_homo   = torch.clamp(1.0 / (2.0 * y_safe) - 1.0 / (2.0 * gamma), min=0.0)
W_hetero = torch.clamp(1.0 / (2.0 * gamma) - 1.0 / (2.0 * y_safe), min=0.0)

W_adaptive = (1.0 - h) * W_homo + h * W_hetero
```

Interpretation:
- `h` acts as an edge-wise heterophily estimate derived from soft-label mismatch.
- For homophilic edges (`h` small), weighting behaves MCP-like (`W_homo`).
- For heterophilic edges (`h` large), it interpolates toward `W_hetero`, reducing the tendency to prune legitimate cross-class structure.

## 2.3 Factory Wiring for This Mode

In `exp/config/get_model.py`, this path is selected via norm `ADAPTIVE`:

```python
# exp/config/get_model.py
elif norm_upper == 'ADAPTIVE':
    w_func = PenaltyFunction.get_w_func('mcp', gamma)
    return _build_rung(w_func, penalty_flag='adaptive'), custom_fit_params
```

So this is a heterophily-aware variant without replacing the base `RUNG` architecture class.

---

## 3) Second (Stronger) Heterophily Change: `RUNG_homophily_adaptive`

The second change introduces a dedicated model class in `model/rung_homophily_adaptive.py`.
This is a deeper change than the adaptive penalty branch.

## 3.1 Architectural Goal

`RUNG_homophily_adaptive` adapts pruning strength per node (and then per edge) based on local soft homophily, instead of using one global threshold behavior.

Core comment in file:

```python
# model/rung_homophily_adaptive.py
# Nodes with low h_i (heterophilic neighborhood) get higher q_i
# (less aggressive pruning)
```

## 3.2 New Local Homophily Estimation

```python
# model/rung_homophily_adaptive.py
P = torch.softmax(F_current, dim=-1)
H = torch.mm(P, P.T)
...
h = sim_sum / neigh_cnt.clamp(min=1.0)
h = h.clamp(min=min_h, max=1.0)
```

This computes per-node soft local homophily `h_i` from class-probability similarity over neighbors.

## 3.3 New Node-Adaptive Percentile Rule

```python
# model/rung_homophily_adaptive.py
def _compute_adaptive_q(self, h: torch.Tensor) -> torch.Tensor:
    q = self.percentile_q + (1.0 - h) * self.q_relax
    return q.clamp(min=self.percentile_q, max=self.q_max)
```

Meaning:
- Lower homophily `h_i` => larger `q_i` => less aggressive thresholding for that node.
- This directly protects heterophilic neighborhoods from over-pruning.

## 3.4 From Node-Specific Quantiles to Edge Thresholds

```python
# model/rung_homophily_adaptive.py
lam_i = gammas / self.scad_a
lam_per_edge = torch.maximum(
    lam_i.unsqueeze(1).expand(N, N),
    lam_i.unsqueeze(0).expand(N, N),
)
W = scad_weight_differentiable(y, lam_per_edge, a=self.scad_a)
```

Differences vs original RUNG:
- Original RUNG uses global scalar threshold behavior through `self.w(y)`.
- `RUNG_homophily_adaptive` builds per-node gamma and per-edge lambda matrices.

## 3.5 Distance Metric Change

Original RUNG uses Euclidean pairwise distances. `RUNG_homophily_adaptive` uses cosine distance:

```python
# model/rung_homophily_adaptive.py
F_unit = F.normalize(F_norm, p=2, dim=-1, eps=self.eps)
cos_sim = torch.mm(F_unit, F_unit.T)
y = (1.0 - cos_sim).clamp(min=0.0, max=2.0)
```

Cosine is scale-invariant and bounded, helping stabilize percentile behavior across layers.

## 3.6 Exact/Fast Paths for Scalability

For larger graphs, a fast approximate node-wise quantile path is used:

```python
# model/rung_homophily_adaptive.py
if N > 500:
    lam_per_edge, gammas = self._compute_per_node_lam_fast(y, A_bool, q_adaptive)
else:
    lam_per_edge, gammas = self._compute_per_node_lam(y, A_bool, q_adaptive)
```

So heterophily adaptation is implemented with performance-aware branching.

---

## 4) Training and CLI Changes for Heterophily Handling

## 4.1 Dedicated Trainer

`RUNG_homophily_adaptive` has its own trainer with diagnostics:

```python
# train_eval_data/fit_homophily_adaptive.py
print(
    f"... h_mean={model._last_h_mean ...:.4f} | "
    f"q_mean={model._last_q_mean ...:.4f}"
)
```

This exposes whether local homophily estimation and adaptive q logic are actually active.

## 4.2 CLI Parameters Added

In `clean.py`, new arguments specific to heterophily adaptation:

```python
# clean.py
parser.add_argument('--q_relax', type=float, default=0.20,
    help='q_i = percentile_q + (1 - h_i) * q_relax.')
parser.add_argument('--q_max', type=float, default=0.99,
    help='Maximum q_i for RUNG_homophily_adaptive.')
parser.add_argument('--homophily_mode', type=str, default='from_F0',
    choices=['from_F0', 'per_layer'])
```

These controls do not exist in original RUNG training flow.

## 4.3 Factory/Dispatch Integration

Model factory branch:

```python
# exp/config/get_model.py
elif model_name == 'RUNG_homophily_adaptive':
    model_ha = RUNG_homophily_adaptive(
        ...,
        percentile_q=percentile_q,
        q_relax=q_relax,
        q_max=q_max,
        homophily_mode=homophily_mode,
        ...,
    )
```

Training dispatch:

```python
# clean.py
elif args.model == 'RUNG_homophily_adaptive':
    fit_homophily_adaptive(cur_model, A, X, y, train_idx, val_idx, **train_param)
```

---

## 5) Dataset-Level Support for Heterophilic Evaluation

The dataset loader explicitly defines heterophilic benchmarks:

```python
# train_eval_data/get_dataset.py
HETEROPHILIC_DATASETS = [
    'chameleon', 'squirrel', 'actor', 'cornell', 'texas', 'wisconsin'
]
```

and routes them through heterophilic-loading utilities in `get_dataset(...)`.

This supports testing the heterophily-specific modifications on appropriate graph families.

---

## 6) Honest Summary: Exactly What Was Changed from Original RUNG

1. Added heterophily-aware adaptive weighting mode inside baseline `RUNG` (`penalty='adaptive'`) via `PenaltyFunction.homophily_adaptive`.
2. Added a dedicated heterophily-optimized model class `RUNG_homophily_adaptive`.
3. Replaced global fixed-like threshold behavior with node-adaptive percentile rules driven by local soft homophily.
4. Converted node thresholds into edge-wise threshold matrices (`lam_per_edge`) before SCAD weighting.
5. Switched distance scoring to cosine inside the dedicated heterophily model.
6. Added fast quantile approximation path for larger graphs.
7. Added heterophily-specific CLI controls (`q_relax`, `q_max`, `homophily_mode`), factory branch, and dedicated trainer diagnostics (`h_mean`, `q_mean`).
8. Added explicit heterophilic dataset handling list and loading path for evaluation.

So in this repository, heterophilic handling is not a single tweak; it is a two-layer strategy:
- lightweight adaptive branch inside original `RUNG`
- stronger dedicated architecture (`RUNG_homophily_adaptive`) with node-adaptive pruning logic.
