# RUNG_combined: Complete Differences from Original RUNG

## Compared Files

- Baseline: `model/rung.py`, `train_eval_data/fit.py`
- Variant: `model/rung_combined.py`, `train_eval_data/fit_combined.py`
- Wiring: `exp/config/get_model.py`

---

## 1) Core Architectural Difference

RUNG_combined merges two ideas simultaneously:
- cosine distance for edge dissimilarity
- percentile-based gamma per layer

```python
# model/rung_combined.py
def _compute_cosine_distance(self, F_norm):
    F_unit = F.normalize(F_norm, p=2, dim=-1, eps=self.eps)
    cos_sim = torch.mm(F_unit, F_unit.T)
    y = (1.0 - cos_sim).clamp(min=0.0, max=2.0)
    return y

def _compute_percentile_lam(self, y, A_bool, q):
    y_edges = y[edge_mask]
    gamma = torch.quantile(y_edges, q)
    lam = gamma / self.scad_a
    return lam
```

---

## 2) Forward Pass Delta vs RUNG

```python
# model/rung_combined.py
F_norm = F / D_sq
y = self._compute_cosine_distance(F_norm)
y = y.detach()
lam_k = self._compute_percentile_lam(y, A_bool, q_k)
W = scad_weight_differentiable(y, lam_k, a=self.scad_a)
```

Compared to RUNG:
- No direct `self.w(y)` usage as in baseline.
- Uses deterministic dynamic threshold from current edge-distance distribution.

---

## 3) Training Delta vs RUNG

It still uses single-group optimizer because no extra learnable gamma/distance module is introduced.

```python
# train_eval_data/fit_combined.py
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

Main differences are richer logging and early-stopping bookkeeping, not optimizer topology.

---

## 4) Factory Difference

```python
# exp/config/get_model.py
elif model_name == 'RUNG_combined':
    model_comb = RUNG_combined(..., percentile_q=percentile_q, use_layerwise_q=use_layerwise_q, percentile_q_late=percentile_q_late, ...)
```

---

## 5) Honest Full Delta vs RUNG

1. Replaces Euclidean pairwise difference usage with cosine distance.
2. Replaces fixed/manual thresholding with percentile-computed threshold each layer.
3. Uses SCAD-style differentiable edge weighting in place of baseline static `w_func` call pattern.
4. Maintains single-group optimizer due no extra trainable threshold module.
5. Keeps the same high-level iterative propagation architecture.
