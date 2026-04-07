# RUNG_parametric_gamma: Complete Differences from Original RUNG

## Compared Files

- Baseline: `model/rung.py`, `train_eval_data/fit.py`
- Variant: `model/rung_parametric_gamma.py`, `train_eval_data/fit_parametric_gamma.py`
- Wiring: `exp/config/get_model.py`, `clean.py`

---

## 1) Core Architectural Difference

RUNG_parametric_gamma replaces fixed penalty thresholding with a 2-parameter geometric threshold schedule across layers.

```python
# model/rung_parametric_gamma.py
self.log_gamma_0 = nn.Parameter(torch.tensor(float(np.log(gamma_0_init)), dtype=torch.float32))
self.raw_decay = nn.Parameter(torch.tensor(logit_init, dtype=torch.float32))

def get_gamma_schedule(self):
    g0 = self.get_gamma_0()
    r  = self.get_decay_rate()
    gammas = []
    for k in range(self.prop_layer_num):
        gk = g0 * torch.pow(r, torch.tensor(float(k), device=g0.device))
        gammas.append(gk)
    return gammas
```

Difference from RUNG:
- RUNG has no learnable gamma schedule
- Variant computes a layer-specific gamma from shared schedule parameters

---

## 2) Forward Pass Delta

### Baseline

```python
# model/rung.py
W = self.w(y)
```

### Variant

```python
# model/rung_parametric_gamma.py
gammas = self.get_gamma_schedule()
for k, gamma_k in enumerate(gammas):
    lam_k = gamma_k / self.scad_a
    W = scad_weight_differentiable(y, lam_k, a=self.scad_a)
```

So the weight function is now dynamic per layer and tied to a learnable schedule.

---

## 3) Training Difference vs RUNG

RUNG uses single-group training; parametric_gamma uses two-group training:

```python
# train_eval_data/fit_parametric_gamma.py
optimizer = torch.optim.Adam([
    {"params": list(model.get_non_gamma_parameters()), "lr": lr, "weight_decay": weight_decay},
    {"params": model.get_gamma_parameters(), "lr": schedule_lr, "weight_decay": 0.0},
])
```

Optional schedule regularization is added:

```python
# train_eval_data/fit_parametric_gamma.py
loss = (
    (model.log_gamma_0 - log_target_gamma_0) ** 2 +
    (model.raw_decay - logit_target_decay) ** 2
)
```

---

## 4) Factory/Dispatch Differences

```python
# exp/config/get_model.py
elif model_name == 'RUNG_parametric_gamma':
    model_pg_param = RUNG_parametric_gamma(..., gamma_0_init=gamma, decay_rate_init=decay_rate_init, ...)
```

```python
# clean.py
elif args.model == 'RUNG_parametric_gamma':
    fit_parametric_gamma(..., gamma_lr_factor=args.gamma_lr_factor, gamma_reg_strength=args.gamma_reg_strength, ...)
```

---

## 5) Honest Full Delta vs RUNG

1. Replaces fixed penalty threshold with learnable geometric threshold schedule.
2. Adds two scalar schedule parameters (`log_gamma_0`, `raw_decay`).
3. Uses differentiable SCAD weighting, not external static `w_func` only.
4. Switches to two-group optimizer training with optional schedule regularization.
5. Keeps original macro-flow (MLP + iterative robust propagation) unchanged.
