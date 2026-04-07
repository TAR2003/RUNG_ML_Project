# RUNG_confidence_lambda: Complete Differences from Original RUNG

## Compared Files

- Baseline: `model/rung.py`, `train_eval_data/fit.py`
- Variant: `model/rung_confidence_lambda.py`, `train_eval_data/fit_confidence_lambda.py`
- Wiring: `exp/config/get_model.py`, `clean.py`

---

## 1) Core Architectural Difference

Original RUNG uses a scalar skip weight (`self.lam`) equally for all nodes.
This variant uses per-node lambda derived from prediction confidence.

```python
# model/rung_confidence_lambda.py
def compute_confidence_lambda(logits_0, lambda_base, alpha, mode='protect_uncertain', normalize=True, eps=1e-6):
    probs = torch.softmax(logits_0, dim=-1)
    conf = probs.max(dim=-1).values
    ...
    raw_lambda = lambda_base * (...) ** alpha
    if normalize:
        lambda_per_node = raw_lambda * (lambda_base / mean_lam)
    return lambda_per_node
```

---

## 2) Forward Pass Delta vs RUNG

### Baseline RUNG update

```python
# model/rung.py
Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)
F = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat
```

### Variant update

```python
# model/rung_confidence_lambda.py
lambda_per_node = compute_confidence_lambda(...)
q = (W * A).sum(-1) / D
q_hat = (q + lambda_per_node).unsqueeze(-1)
F = (W * A_tilde) @ F / q_hat + lambda_per_node.unsqueeze(-1) * F0 / q_hat
```

Main change:
- scalar lambda in baseline -> vector lambda per node

---

## 3) Additional Parameter Delta

Variant adds alpha sharpness control for confidence mapping:

```python
# model/rung_confidence_lambda.py
self.raw_alpha = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

@property
def alpha(self):
    return (nnF.softplus(self.raw_alpha) + 0.5).item()
```

Also retains per-layer gamma parameters inherited from learnable-gamma style design.

---

## 4) Training Differences vs RUNG

Training in `fit_confidence_lambda.py` is much richer:

1. Three optimizer groups

```python
optimizer = torch.optim.Adam([
    {'params': list(model.get_non_gamma_alpha_parameters()), 'lr': lr, 'weight_decay': weight_decay},
    {'params': list(model.get_gamma_parameters()), 'lr': lr * gamma_lr_factor, 'weight_decay': 0.0},
    {'params': list(model.get_alpha_parameters()), 'lr': lr * alpha_lr_factor, 'weight_decay': 0.0},
])
```

2. Warmup stage (freeze gamma and alpha first)

```python
if warmup_epochs > 0:
    for p in model.get_gamma_parameters():
        p.requires_grad_(False)
    for p in model.get_alpha_parameters():
        p.requires_grad_(False)
```

3. Lambda-confidence correlation diagnostics and alpha regularization.

This is a large departure from base RUNG's simple one-group CE training.

---

## 5) Factory/Dispatch Differences

```python
# exp/config/get_model.py
elif model_name == 'RUNG_confidence_lambda':
    model_cl = RUNG_confidence_lambda(...,
        gamma_init=gamma,
        gamma_init_strategy=gamma_init_strategy,
        alpha_init=alpha_init,
        confidence_mode=confidence_mode,
        normalize_lambda=normalize_lambda,
    )
```

```python
# clean.py
elif args.model == 'RUNG_confidence_lambda':
    fit_confidence_lambda(...,
        gamma_lr_factor=args.gamma_lr_factor,
        alpha_lr_factor=args.alpha_lr_factor,
        gamma_reg_strength=args.gamma_reg_strength,
        alpha_reg_strength=args.alpha_reg_strength,
        warmup_epochs=args.warmup_epochs,
    )
```

---

## 6) Honest Full Delta vs RUNG

1. Replaces scalar skip term with confidence-conditioned per-node skip term.
2. Adds learnable alpha for confidence-to-lambda mapping sharpness.
3. Uses learnable per-layer SCAD gamma mechanism.
4. Adds three-group optimizer and warmup/unfreeze schedule.
5. Adds mechanism diagnostics (lambda spread and confidence correlation).
6. Retains core MLP + iterative robust propagation skeleton.
