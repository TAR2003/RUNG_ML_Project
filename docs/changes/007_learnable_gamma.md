# 007 — Per-Layer Learnable Gamma (`RUNG_learnable_gamma`)

## Date
2026-03-03

## Parent Model
`RUNG_new_SCAD` (which is `RUNG` with SCAD penalty, `norm=SCAD`)

## New Model Name
`RUNG_learnable_gamma`

---

## Core Hypothesis

Feature differences

```
y_ij^(k) = || f_i^(k) / sqrt(d_i)  −  f_j^(k) / sqrt(d_j) ||_2
```

shrink systematically across aggregation layers as features become smoother.
A single fixed SCAD threshold `gamma` is suboptimal at all but one layer depth.
Making `gamma` learnable per layer allows each layer to find its own optimal
threshold through gradient descent.

---

## What Changed vs `RUNG_new_SCAD`

| Aspect                 | `RUNG_new_SCAD`               | `RUNG_learnable_gamma`                        |
|------------------------|-------------------------------|-----------------------------------------------|
| SCAD threshold `lam`   | Fixed scalar (`gamma / a`)    | K learnable scalars `lam^(0..K-1)`             |
| Parameterisation       | `lam` captured in closure     | `lam^(k) = exp(log_lam^(k))`                  |
| Parameter count        | 0 threshold params            | K threshold params (e.g. 10 for K=10)         |
| Optimizer              | Single Adam group             | Two Adam groups (separate LR for gamma)        |
| Gamma LR               | N/A                           | `lr × gamma_lr_factor` (default 0.2×)         |
| SCAD implementation    | Boolean indexing (non-diffble) | `torch.where` (fully differentiable)          |
| Model interface        | `forward(A, X)`               | `forward(A, X)` (identical)                   |

---

## Files Created

| File | Purpose |
|------|---------|
| `model/rung_learnable_gamma.py` | Model class + differentiable SCAD penalty |
| `train_eval_data/fit_learnable_gamma.py` | Two-group optimizer training loop |
| `experiments/analyze_learned_gammas.py` | Convergence and distribution plots |
| `docs/changes/007_learnable_gamma.md` | This file |

## Files Modified

| File | Change |
|------|--------|
| `exp/config/get_model.py` | Added `RUNG_learnable_gamma` factory branch; imported new class |
| `clean.py` | Added import, argparse args, `RUNG_learnable_gamma` training dispatch |

---

## Mathematical Justification

### SCAD weight function (differentiable form)

```
W_ij = dρ_SCAD(y_ij) / dy²

     = 1 / (2 y)                          if y < lam          [region 1]
       (a·lam − y) / ((a−1)·lam · 2·y)   if lam ≤ y < a·lam  [region 2]
       0                                   if y ≥ a·lam        [region 3]
```

where `lam^(k) = exp(log_lam^(k))` is the learnable threshold for layer `k`.

### Gradient flow

Backprop path for layer k:

```
Loss  →  F^(K)  →  ...  →  W^(k) = f(y^(k), lam^(k))  →  lam^(k) = exp(θ^(k))  →  θ^(k)
```

Gradient of W w.r.t. `lam`:

```
dW_ij / d(lam):
    Region 1 (y < lam):       dW/dlam = 0            (W doesn't depend on lam)
    Region 2 (lam ≤ y < a·lam): dW/dlam = −a / ((a−1) · lam · 2y)   (negative)
    Region 3 (y ≥ a·lam):     dW/dlam = 0            (W = 0 everywhere)
```

Only **region-2 edges** (the SCAD transition zone) carry gradient signal.
This means `lam^(k)` converges to the value where some edges sit in the
transition zone — i.e., `lam` naturally finds the tail of the feature
difference distribution at each layer.

**Implementation note:** `scad_weight_differentiable()` uses `torch.where`
throughout.  The old `PenaltyFunction.scad()` uses boolean indexing
(`W[mask] = ...`) which breaks autograd w.r.t. `lam`.  Do **not** replace
the new implementation with the old one.

---

## Naming Convention

```
CLI --gamma G  →  lam_init = G / a    (matches RUNG_new_SCAD convention)
                  gamma (zero-cutoff) = a × lam
```

With default `a = 3.7`:
- `--gamma 6.0` → `lam_init = 6.0 / 3.7 ≈ 1.622`
- `get_learned_gammas()` returns `a × lam` in the same scale as `--gamma`

---

## How to Run

### Train (clean)

```bash
# Default settings — matches RUNG_new_SCAD default gamma
python clean.py --model RUNG_learnable_gamma --data cora --gamma 6.0

# Decreasing initialisation (theoretically motivated)
python clean.py --model RUNG_learnable_gamma --data cora \
                --gamma 6.0 --gamma_init_strategy decreasing

# Lower gamma LR for more conservative updates
python clean.py --model RUNG_learnable_gamma --data cora \
                --gamma_lr_factor 0.1

# With gamma regularisation (if gammas diverge)
python clean.py --model RUNG_learnable_gamma --data cora \
                --gamma_reg_strength 0.01
```

### Attack (evasion)

```bash
# Uses pre-trained model saved by clean.py
python attack.py --model RUNG_learnable_gamma --data cora --gamma 6.0
```

### Analysis

```bash
# Quick demo: trains on Cora and produces convergence / distribution plots
python experiments/analyze_learned_gammas.py --data cora --gamma 6.0

# With decreasing initialisation
python experiments/analyze_learned_gammas.py \
    --data cora --gamma 6.0 --gamma_init_strategy decreasing
```

---

## Ablation Experiments

| Ablation | Values to test | Metric |
|----------|---------------|--------|
| `gamma_lr_factor` | 0.05, 0.1, **0.2**, 0.5, 1.0 | val acc, gamma convergence |
| `gamma_init_strategy` | **uniform**, decreasing, increasing | val acc, final gamma pattern |
| `gamma_reg_strength` | 0, 0.001, **0.01**, 0.1 | val acc, gamma stability |
| `gamma_init` | 3.0, **6.0**, 9.0 | val acc, convergence speed |

Bold = recommended default.

---

## Expected Behaviour

| Observation | Interpretation |
|-------------|---------------|
| Learned gammas **decrease** with layer depth | Feature diffs shrink as expected → per-layer gamma is working correctly |
| Gammas all converge to **same value** | Gradient signal is too weak; increase `gamma_lr_factor` |
| Gammas **diverge** (NaN or very large) | Gradient too large; decrease `gamma_lr_factor` or add `gamma_reg_strength` |
| Gammas stay near initialisation | Model is stuck; try `decreasing` initialisation |

---

## Results

*Fill in after running experiments.*

| Budget | RUNG (base) | RUNG_new_SCAD | RUNG_learnable_gamma |
|--------|-------------|---------------|----------------------|
| 0 %    |             |               |                      |
| 5 %    |             |               |                      |
| 10 %   |             |               |                      |
| 20 %   |             |               |                      |
| 30 %   |             |               |                      |
| 40 %   |             |               |                      |

## Learned Gamma Values (after training)

*Fill in after running `model.log_gamma_stats()`.*

## Interpretation

*Fill in after analysis — do gammas decrease with depth? by how much?*
