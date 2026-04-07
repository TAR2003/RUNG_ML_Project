# RUNG_percentile_gamma: Complete Differences from Original RUNG

## Compared Files

- Baseline: `model/rung.py`, `train_eval_data/fit.py`
- Variant: `model/rung_percentile_gamma.py`, `train_eval_data/fit_percentile_gamma.py`
- Wiring: `exp/config/get_model.py`, `clean.py`

---

## 1) Core Architectural Difference

Instead of fixed/manual gamma in RUNG, this model computes gamma from edge-distance percentiles at each layer.

```python
# model/rung_percentile_gamma.py
def _compute_percentile_lam(self, y, A_bool, q):
    edge_mask = A_bool & ~eye_bool
    y_edges = y[edge_mask]
    gamma = torch.quantile(y_edges, q)
    lam   = gamma / self.scad_a
    return lam
```

Baseline RUNG has no percentile computation; it directly applies `self.w(y)`.

---

## 2) Forward Pass Delta

```python
# model/rung_percentile_gamma.py
q_k = self._layer_q_values[k]
Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
y = Z.sqrt()
lam_k = self._compute_percentile_lam(y, A_bool, q_k)
W = scad_weight_differentiable(y, lam_k, a=self.scad_a)
```

Difference from RUNG:
- Layer-adaptive threshold from data distribution
- SCAD weighting via `scad_weight_differentiable`

---

## 3) New Hyperparameter Pattern

In this model, threshold control is by percentile values (`percentile_q`, optional `percentile_q_late`) instead of fixed `gamma`.

```python
# model/rung_percentile_gamma.py
self._layer_q_values = [self._get_q_for_layer(k) for k in range(prop_step)]
```

---

## 4) Training Difference vs RUNG

Training stays single-group (because gamma is not a trainable parameter), but adds percentile-gamma logging.

```python
# train_eval_data/fit_percentile_gamma.py
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
...
if log_gamma_every > 0 and (epoch + 1) % log_gamma_every == 0:
    gammas = model.get_last_gammas()
```

Compared to RUNG fit:
- Still one optimizer group
- More diagnostics for gamma dynamics

---

## 5) Factory/Dispatch Differences

```python
# exp/config/get_model.py
elif model_name == 'RUNG_percentile_gamma':
    model_pg = RUNG_percentile_gamma(..., percentile_q=percentile_q, use_layerwise_q=use_layerwise_q, percentile_q_late=percentile_q_late, ...)
```

```python
# clean.py
elif args.model == 'RUNG_percentile_gamma':
    fit_percentile_gamma(cur_model, A, X, y, train_idx, val_idx, **train_param)
```

---

## 6) Honest Full Delta vs RUNG

1. Replaces fixed/manual threshold with quantile-based threshold each layer.
2. Supports optional early/late layer percentile split.
3. Keeps single-group training (no learnable gamma parameters).
4. Adds gamma-history diagnostics.
5. Preserves same macro architecture and QN-IRLS skeleton.
