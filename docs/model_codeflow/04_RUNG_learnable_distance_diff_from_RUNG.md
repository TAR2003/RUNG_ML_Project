# RUNG_learnable_distance: Complete Differences from Original RUNG

## Compared Files

- Baseline: `model/rung.py`, `train_eval_data/fit.py`
- Variant: `model/rung_learnable_distance.py`, `train_eval_data/fit_learnable_distance.py`
- Wiring: `exp/config/get_model.py`, `clean.py`

---

## 1) Core Architectural Difference

RUNG uses Euclidean pairwise differences only.
This variant introduces a pluggable distance module: cosine, projection, or bilinear.

```python
# model/rung_learnable_distance.py
class DistanceModule(nn.Module):
    ...
    if self.mode == 'cosine':
        y = 1.0 - cos_sim
    elif self.mode == 'projection':
        H = self.proj(F_norm)
        ...
    elif self.mode == 'bilinear':
        H = self.W(F_norm)
        ...
```

---

## 2) Forward Pass Delta vs RUNG

### Baseline RUNG

```python
# model/rung.py
Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
y = Z.sqrt()
W = self.w(y)
```

### Variant

```python
# model/rung_learnable_distance.py
F_norm = F / D_sq
y = self.distance(F_norm)
if self.distance.count_parameters() == 0:
    y = y.detach()
lam_k = self._compute_percentile_lam(y, A_bool, q_k)
W = scad_weight_differentiable(y, lam_k, a=self.scad_a)
```

Two key differences:
1. Distance metric can be learnable or fixed cosine.
2. Threshold still percentile-adaptive (inherits percentile-gamma idea).

---

## 3) Training Differences vs RUNG

This variant can train with one or two optimizer groups depending on distance mode.

```python
# train_eval_data/fit_learnable_distance.py
if len(dist_params) == 0:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

return torch.optim.Adam([
    {'params': main_params, 'lr': lr, 'weight_decay': weight_decay},
    {'params': dist_params, 'lr': lr * dist_lr_factor, 'weight_decay': 0.0},
])
```

Baseline RUNG never uses this conditional two-group behavior.

---

## 4) Factory/Dispatch Differences

```python
# exp/config/get_model.py
elif model_name == 'RUNG_learnable_distance':
    model_ld = RUNG_learnable_distance(..., percentile_q=percentile_q, distance_mode=distance_mode, proj_dim=proj_dim, ...)
```

```python
# clean.py
elif args.model == 'RUNG_learnable_distance':
    fit_learnable_distance(..., dist_lr_factor=args.dist_lr_factor, ...)
```

---

## 5) Honest Full Delta vs RUNG

1. Replaces fixed Euclidean distance with configurable distance module.
2. Adds optional learnable distance parameters (projection/bilinear modes).
3. Uses percentile-based SCAD thresholding, not fixed `w_func` only.
4. Training optimizer can become two-group when distance module is learnable.
5. Retains base MLP + robust iterative propagation structure.
