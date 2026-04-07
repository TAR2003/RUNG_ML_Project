# RUNG_learnable_combined: Complete Differences from Original RUNG

## Compared Files

- Baseline: `model/rung.py`, `train_eval_data/fit.py`
- Variant: `model/rung_learnable_combined.py`, `train_eval_data/fit_learnable_combined.py`
- Wiring: `exp/config/get_model.py`, `clean.py`

---

## 1) Core Architectural Difference

This variant combines cosine distance with learnable gamma constrained to cosine-scale range.

```python
# model/rung_learnable_combined.py
class CosineLearnableGamma(nn.Module):
    if gamma_mode == 'per_layer':
        self.raw_gamma = nn.Parameter(torch.zeros(prop_step))
    else:
        self.raw_g0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.raw_decay = nn.Parameter(torch.tensor(logit_085, dtype=torch.float32))

    def get_gamma(self, layer_idx):
        if self.gamma_mode == 'per_layer':
            return torch.sigmoid(self.raw_gamma[layer_idx]) * 2.0
        gamma_0 = torch.sigmoid(self.raw_g0) * 2.0
        decay_rate = torch.sigmoid(self.raw_decay)
        return gamma_0 * torch.pow(decay_rate, k)
```

Baseline RUNG has no such gamma module and no [0,2]-bounded gamma mapping.

---

## 2) Forward Pass Delta vs RUNG

```python
# model/rung_learnable_combined.py
F_norm = F / D_sq
y = self.distance(F_norm).detach()  # cosine distance
gamma_k = self.gamma_module.get_gamma(k)
lam_k = (gamma_k / self.scad_a).clamp(min=self.eps)
W = scad_weight_differentiable(y, lam_k, a=self.scad_a)
```

Key differences:
1. Distance is cosine-based (through `DistanceModule(..., mode='cosine')`).
2. Gamma is explicitly learnable and constrained through sigmoid parameterization.

---

## 3) Training Differences vs RUNG

Uses two optimizer groups (main parameters and gamma parameters):

```python
# train_eval_data/fit_learnable_combined.py
return torch.optim.Adam([
    {'params': model.get_non_gamma_parameters(), 'lr': lr, 'weight_decay': weight_decay},
    {'params': model.get_gamma_parameters(), 'lr': lr * gamma_lr_factor, 'weight_decay': 0.0},
])
```

Baseline RUNG does single-group optimizer only.

---

## 4) Factory/Dispatch Differences

```python
# exp/config/get_model.py
elif model_name == 'RUNG_learnable_combined':
    model_lc = RUNG_learnable_combined(..., gamma_mode=gamma_mode, ...)
```

```python
# clean.py
elif args.model == 'RUNG_learnable_combined':
    fit_learnable_combined(..., gamma_lr_factor=args.gamma_lr_factor, ...)
```

---

## 5) Honest Full Delta vs RUNG

1. Adds cosine-distance path for edge suspiciousness.
2. Adds learnable gamma module with two modes (`per_layer` or schedule).
3. Constrains gamma values to cosine distance scale via sigmoid mapping.
4. Uses two-group optimizer for gamma control.
5. Keeps base propagation skeleton and MLP backbone structure.
