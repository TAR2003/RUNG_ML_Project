# RUNG_homophily_adaptive: Complete Differences from Original RUNG

## Compared Files

- Baseline: `model/rung.py`, `train_eval_data/fit.py`
- Variant: `model/rung_homophily_adaptive.py`, `train_eval_data/fit_homophily_adaptive.py`
- Wiring: `exp/config/get_model.py`, `clean.py`

---

## 1) Core Architectural Difference

This model makes gamma-node-specific using local soft homophily estimates.

```python
# model/rung_homophily_adaptive.py
def _compute_soft_homophily(self, F_current, A_bool):
    P = torch.softmax(F_current, dim=-1)
    H = torch.mm(P, P.T)
    ...
    h = sim_sum / neigh_cnt.clamp(min=1.0)
    return h

def _compute_adaptive_q(self, h):
    q = self.percentile_q + (1.0 - h) * self.q_relax
    return q.clamp(min=self.percentile_q, max=self.q_max)
```

RUNG baseline has no per-node homophily computation and no adaptive percentile per node.

---

## 2) Per-node Gamma Construction (Main Delta)

```python
# model/rung_homophily_adaptive.py
lam_per_edge, gammas = self._compute_per_node_lam_fast(y, A_bool, q_adaptive)
W = scad_weight_differentiable(y, lam_per_edge, a=self.scad_a)
```

This is a major difference from RUNG:
- RUNG uses a single global weighting function over y.
- This variant uses per-node/per-edge thresholding derived from local graph context.

---

## 3) Forward Pass Delta

```python
# model/rung_homophily_adaptive.py
if self.homophily_mode == 'from_F0':
    h = self._compute_soft_homophily(F0, A_bool)
    q_adaptive = self._compute_adaptive_q(h)
...
y = self._compute_cosine_distance(F_norm).detach()
...
F = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat
```

Structure remains RUNG-like, but y/gamma generation is fundamentally changed.

---

## 4) Training Difference vs RUNG

Training remains single-group Adam, but with model-specific diagnostics:

```python
# train_eval_data/fit_homophily_adaptive.py
print(f"... h_mean={model._last_h_mean:.4f} | q_mean={model._last_q_mean:.4f}")
```

No gamma-parameter optimizer group is used because adaptive q/gamma are data-derived.

---

## 5) Factory/Dispatch Differences

```python
# exp/config/get_model.py
elif model_name == 'RUNG_homophily_adaptive':
    model_ha = RUNG_homophily_adaptive(..., percentile_q=percentile_q, q_relax=q_relax, q_max=q_max, homophily_mode=homophily_mode, ...)
```

```python
# clean.py
elif args.model == 'RUNG_homophily_adaptive':
    fit_homophily_adaptive(cur_model, A, X, y, train_idx, val_idx, **train_param)
```

---

## 6) Honest Full Delta vs RUNG

1. Adds soft local homophily estimation per node.
2. Converts homophily into node-adaptive percentile targets.
3. Builds per-node (then per-edge) SCAD thresholds instead of a global one.
4. Uses cosine distance in place of baseline Euclidean path.
5. Keeps original iterative propagation update equation pattern after weight construction.
