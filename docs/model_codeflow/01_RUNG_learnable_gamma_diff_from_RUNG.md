# RUNG_learnable_gamma: Complete Differences from Original RUNG

## Compared Files

- Baseline: `model/rung.py`, `train_eval_data/fit.py`
- Variant: `model/rung_learnable_gamma.py`, `train_eval_data/fit_learnable_gamma.py`
- Wiring: `exp/config/get_model.py`, `clean.py`

---

## 1) Core Architectural Difference

Original RUNG uses a fixed edge-threshold behavior via `w_func` (for example MCP with fixed gamma).

RUNG_learnable_gamma replaces fixed thresholding with per-layer learnable SCAD thresholds.

### Baseline snippet

```python
# model/rung.py
if self.penalty == 'adaptive':
    ...
else:
    W = self.w(y)
```

### Variant snippet

```python
# model/rung_learnable_gamma.py
self.log_lams = nn.ParameterList([
    nn.Parameter(torch.tensor(float(np.log(l)), dtype=torch.float32))
    for l in lam_inits
])

for k, log_lam_k in enumerate(self.log_lams):
    lam_k = torch.exp(log_lam_k)
    ...
    W = scad_weight_differentiable(y, lam_k, a=self.scad_a)
```

What changed:
- RUNG: one fixed penalty function via `self.w`
- learnable_gamma: K learnable scalars (`log_lams`) controlling K layer-specific thresholds

---

## 2) New Differentiable SCAD Weight Function

RUNG_learnable_gamma adds a differentiable SCAD derivative implementation to preserve gradients through threshold parameters.

```python
# model/rung_learnable_gamma.py
def scad_weight_differentiable(y, lam, a=3.7, eps=1e-8):
    y_safe = y.clamp(min=eps)
    region1_val = 1.0 / (2.0 * y_safe)
    denom2 = ((a - 1.0) * lam * 2.0 * y_safe).clamp(min=eps)
    region2_val = (a * lam - y) / denom2
    ...
    W = torch.where(in_region1, region1_val,
                    torch.where(in_region2, region2_val, region3_val))
```

Difference from RUNG:
- Baseline does not own this function in-model; it relies on external `w_func`
- Variant makes threshold computation first-class and trainable

---

## 3) Forward Pass Differences

### Baseline

```python
# model/rung.py
Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
y = Z.sqrt()
W = self.w(y)
```

### Variant

```python
# model/rung_learnable_gamma.py
Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
y = Z.sqrt()
W = scad_weight_differentiable(y, lam_k, a=self.scad_a)
```

Additional details in variant:
- Zero-diagonal operation is done out-of-place (`W = W * (1.0 - eye)`) to avoid autograd break on threshold parameters

---

## 4) Optimizer and Training Differences

Baseline RUNG uses single-group Adam in `train_eval_data/fit.py`.

Variant uses two groups in `train_eval_data/fit_learnable_gamma.py`:

```python
optimizer = torch.optim.Adam([
    {"params": list(model.get_non_gamma_parameters()), "lr": lr, "weight_decay": weight_decay},
    {"params": list(model.get_gamma_parameters()), "lr": gamma_lr, "weight_decay": 0.0},
])
```

Also adds optional threshold regularization:

```python
# train_eval_data/fit_learnable_gamma.py
reg_loss = sum((log_lam - log_target) ** 2 for log_lam in model.log_lams)
```

And periodic gamma logging.

---

## 5) Factory/Dispatch Differences

Model creation branch:

```python
# exp/config/get_model.py
elif model_name == 'RUNG_learnable_gamma':
    model_lg = RUNG_learnable_gamma(..., gamma_init=gamma, gamma_init_strategy=gamma_init_strategy, ...)
```

Training dispatch:

```python
# clean.py
elif args.model == 'RUNG_learnable_gamma':
    fit_learnable_gamma(..., gamma_lr_factor=args.gamma_lr_factor, gamma_reg_strength=args.gamma_reg_strength, ...)
```

---

## 6) Honest Summary of All Differences vs RUNG

1. Fixed penalty call `self.w(y)` is replaced by layer-wise learnable SCAD thresholds.
2. Adds K trainable threshold parameters (`log_lams`).
3. Adds differentiable SCAD weighting function in model file.
4. Training changes from one optimizer group to two groups.
5. Adds optional gamma regularization and explicit gamma diagnostics.
6. Keeps the same high-level structure: MLP -> robust propagation loop -> logits.
