# RUNG Base Model Code Flow (Source of Truth)

This document describes the original RUNG pipeline as implemented in this repository.
It is based directly on code in the files listed below.

## Primary Files

- `clean.py` (experiment/training entry point)
- `exp/config/get_model.py` (model factory and hyperparameter routing)
- `model/rung.py` (core RUNG architecture and propagation)
- `model/penalty.py` and `model/att_func.py` (edge weighting functions)
- `train_eval_data/fit.py` (standard training loop for base RUNG)
- `utils.py` (graph/math helpers: loops, symmetric normalization, pairwise distances)

---

## 1) High-Level Execution Path

### 1.1 CLI to Model Build

In `clean.py`, command-line args are parsed and then model-specific config is assembled.
For base RUNG, the branch is in `exp/config/get_model.py` under `if model_name == 'RUNG':`.

```python
# exp/config/get_model.py
if model_name == 'RUNG':
    norm_upper = norm.upper()
    if norm_upper == 'MCP':
        return _build_rung(get_mcp_att_func(gamma)), custom_fit_params
    elif norm_upper == 'SCAD':
        lam_scad = gamma / 3.7
        return _build_rung(get_scad_att_func(lam_scad, 3.7)), custom_fit_params
    elif norm_upper == 'L1':
        return _build_rung(get_l12_att_func('L1')), custom_fit_params
    elif norm_upper == 'L2':
        return _build_rung(get_l12_att_func('L2')), custom_fit_params
    elif norm_upper == 'ADAPTIVE':
        return _build_rung(get_mcp_att_func(gamma), penalty_flag='adaptive'), custom_fit_params
```

### 1.2 Training Dispatch

In `clean.py`, base RUNG uses the generic fit routine:

```python
# clean.py
elif args.model in ('RUNG', 'RUNG_new', 'MLP', 'L1', 'APPNP'):
    fit(cur_model, A, X, y, train_idx, val_idx, **train_param)
```

### 1.3 Core Model Forward

`model/rung.py` runs:
1. MLP feature-to-logit projection (`F0 = self.mlp(F)`)
2. Graph preprocessing (`add_loops`, degree vector, symmetric normalization)
3. K propagation steps (QN-IRLS by default)

```python
# model/rung.py
F0 = self.mlp(F)
A = add_loops(A)
D = A.sum(-1)
D_sq = D.sqrt().unsqueeze(-1)
A_tilde = sym_norm(A)

for _layer in range(self.prop_layer_num):
    Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
    y = Z.sqrt()
    ...
```

---

## 2) Exact Base RUNG Computation Flow

## 2.1 Edge Difference Matrix

RUNG computes pairwise normalized feature differences:

```python
# model/rung.py
Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
y = Z.sqrt()
```

The helper comes from `utils.py`:

```python
# utils.py
def pairwise_squared_euclidean(X, Y):
    squared_X_feat_norms = (X * X).sum(dim=-1)
    squared_Z_feat_norms = (Y * Y).sum(dim=-1)
    pairwise_feat_dot_prods = X @ Y.transpose(-2, -1)
    return (-2 * pairwise_feat_dot_prods + squared_X_feat_norms[:, None] + squared_Z_feat_norms[None, :]).clamp_min(0)
```

## 2.2 Penalty Weighting

RUNG converts differences into IRLS weights via `self.w(y)` or adaptive mode:

```python
# model/rung.py
if self.penalty == 'adaptive':
    soft_labels = torch.softmax(F.detach(), dim=-1)
    W = PenaltyFunction.homophily_adaptive(y, self.gamma, soft_labels)
else:
    W = self.w(y)
```

For default MCP behavior, the core formula in `model/penalty.py` is:

```python
# model/penalty.py
@staticmethod
def mcp(y: torch.Tensor, gamma: float, eps: float = 1e-8) -> torch.Tensor:
    y_safe = y.clamp(min=eps)
    W = torch.clamp(1.0 / (2.0 * y_safe) - 1.0 / (2.0 * gamma), min=0.0)
    return W
```

## 2.3 Weight Sanitization

Diagonal edges are removed and NaNs are guarded:

```python
# model/rung.py
idx = torch.arange(W.shape[0], device=W.device)
W[idx, idx] = 0.0
W[torch.isnan(W)] = 1.0
```

## 2.4 QN-IRLS Update

The default update is quasi-Newton IRLS:

```python
# model/rung.py
Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)
F = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat
```

Where `self.lam = 1.0 / lam_hat - 1.0`.

---

## 3) Training Flow for Base RUNG

Base RUNG uses `train_eval_data/fit.py`:

```python
# train_eval_data/fit.py
optimizer = torch.optim.Adam(model.parameters(), **{key: kwargs[key] for key in kwargs if key in ['lr', 'weight_decay']})

for i in tqdm.trange(kwargs['max_epoch'] if 'max_epoch' in kwargs else 3000):
    model.train()
    optimizer.zero_grad()
    loss = F.cross_entropy(model(A, X)[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        model.eval()
        print(accuracy(model(A, X)[val_idx], y[val_idx]))
```

Key properties:
- Single optimizer group
- Standard cross-entropy objective
- No special gamma/lambda scheduler logic
- No custom regularizer on RUNG internals

---

## 4) Factory-Level Model Parameters (Base RUNG)

RUNG in this repo is built through `_build_rung(...)` with:
- hidden dims fixed to `[64]`
- `lam_hat=0.9` unless overridden by caller
- norm branch controls the weight function (`MCP`, `SCAD`, `L1`, `L2`, `ADAPTIVE`)

```python
# exp/config/get_model.py
def _build_rung(w_func, penalty_flag=None):
    return RUNG(
        D, C, [64], w_func, 0.9,
        penalty=penalty_flag,
        gamma=gamma,
    ).to(device)
```

---

## 5) Minimal Conceptual Dataflow

1. `X -> MLP -> F0`
2. Build normalized graph operators from `A`
3. Repeat K times:
   - Compute pairwise differences from normalized features
   - Convert differences to robust edge weights
   - Apply QN-IRLS propagation with skip term toward `F0`
4. Return final propagated logits

This is the exact baseline that all `rung_*` variants in this repo modify.
