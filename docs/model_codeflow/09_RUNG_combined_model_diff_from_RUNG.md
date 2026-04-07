# RUNG_combined_model: Complete Differences from Original RUNG

## Compared Files

- Baseline: `model/rung.py`, `train_eval_data/fit.py`
- Variant: `model/rung_combined_model.py`, `train_eval_data/fit_combined_model.py`
- Wiring: `exp/config/get_model.py`, `clean.py`

---

## 1) Core Architectural Difference

This is the most composite variant in the repository.
It combines:
1. cosine distance
2. parametric gamma schedule
3. percentile gamma
4. learnable blending between parametric and percentile gamma

```python
# model/rung_combined_model.py
gamma^(k) = alpha * gamma_param^(k) + (1-alpha) * gamma_data^(k)
alpha = sigmoid(raw_alpha_blend)
```

---

## 2) New Parameters vs RUNG

```python
# model/rung_combined_model.py
self.log_gamma_0 = nn.Parameter(torch.tensor(float(np.log(gamma_0_init))))
self.raw_decay = nn.Parameter(torch.tensor(logit_decay))
self.raw_alpha_blend = nn.Parameter(torch.tensor(logit_alpha))
```

RUNG baseline has none of these schedule/blend parameters.

---

## 3) Layer-Level Delta (RUNGLayer_combined_model)

```python
# model/rung_combined_model.py (RUNGLayer_combined_model.forward)
F_unit = F.normalize(F_norm, p=2, dim=-1, eps=self.eps)
cos_sim = torch.mm(F_unit, F_unit.T)
y = (1.0 - cos_sim).clamp(min=0.0, max=2.0)

gamma_data_k = torch.quantile(y_edges, self.percentile_q)
lam_data_k = gamma_data_k / self.scad_a
lam_param_k = gamma_param_k / self.scad_a
lam_combined_k = alpha_blend * lam_param_k + (1.0 - alpha_blend) * lam_data_k
W = scad_weight_differentiable(y, lam_combined_k, a=self.scad_a)
```

So relative to RUNG, both distance metric and threshold-generation logic are replaced.

---

## 4) Training Differences vs RUNG

Two-group optimizer where schedule+blend parameters are isolated with reduced LR:

```python
# train_eval_data/fit_combined_model.py
return torch.optim.Adam([
    {'params': model.get_non_gamma_parameters(), 'lr': lr, 'weight_decay': weight_decay},
    {'params': model.get_gamma_parameters(), 'lr': lr * gamma_lr_factor, 'weight_decay': 0.0},
])
```

Optional regularizer also includes alpha blend target:

```python
# train_eval_data/fit_combined_model.py
loss = (
    (model.log_gamma_0 - log_target_gamma_0) ** 2 +
    (model.raw_decay - logit_target_decay) ** 2 +
    (model.raw_alpha_blend - logit_target_alpha_blend) ** 2
)
```

Baseline RUNG does none of this.

---

## 5) Factory/Dispatch Differences

```python
# exp/config/get_model.py
elif model_name == 'RUNG_combined_model':
    model_cmb = RUNG_combined_model(...,
        percentile_q=percentile_q,
        gamma_0_init=gamma,
        decay_rate_init=decay_rate_init,
        alpha_blend_init=alpha_blend_init,
        ...)
```

```python
# clean.py
elif args.model == 'RUNG_combined_model':
    fit_combined_model(...,
        gamma_lr_factor=args.gamma_lr_factor,
        gamma_reg_strength=args.gamma_reg_strength,
        ...)
```

---

## 6) Honest Full Delta vs RUNG

1. Switches to cosine-distance edge scoring.
2. Adds learnable parametric gamma schedule (gamma_0 + decay).
3. Adds data-driven percentile gamma path.
4. Adds learnable alpha to blend schedule gamma and percentile gamma.
5. Uses dedicated optimizer grouping and optional schedule/blend regularization.
6. Keeps base top-level idea: MLP backbone followed by iterative robust propagation.
