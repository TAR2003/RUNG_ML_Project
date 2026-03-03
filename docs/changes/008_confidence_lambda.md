# 008 — Per-Node Confidence-Weighted Lambda (RUNG_confidence_lambda)

## Date
2026-03-03

## Parent Model
RUNG_learnable_gamma

## New Model Name
RUNG_confidence_lambda

## Core Hypothesis
Adversarial edges are preferentially added to nodes with low prediction
confidence (nodes near the decision boundary where flipping a few edges
has the largest impact on the prediction).  Those nodes receive the most
corrupted neighbourhood signals, but with a fixed scalar lambda every node
has the same skip-connection weight.  Per-node confidence-weighted lambda
allocates each node a skip-connection strength that reflects its own
prediction reliability, adapting the model to per-node attack vulnerability.

## What Changed vs RUNG_learnable_gamma

| Aspect             | RUNG_learnable_gamma         | RUNG_confidence_lambda                    |
|--------------------|------------------------------|-------------------------------------------|
| Lambda type        | Scalar float (fixed)         | Vector [N] computed from confidence       |
| Lambda formula     | λ = 1/λ̂ − 1                 | λ_i = λ_base · g(conf_i, α, mode)        |
| New parameters     | 0                            | 1 (raw_alpha sharpness)                   |
| Optimizer groups   | 2 (main + gamma)             | 3 (main + gamma + alpha)                  |
| Training phases    | 1 (joint from epoch 1)       | 2 (warmup MLP, then joint)                |
| QN-IRLS update     | scalar λ in denominator      | per-node λ_i in denominator               |

### QN-IRLS Update (Matrix Form)

**RUNG_learnable_gamma (Eq. 8 original):**
```
F^(k+1) = (diag(q^(k)) + λI)^{-1} [(W^(k) ⊙ Ã)F^(k) + λF^(0)]
```

**RUNG_confidence_lambda (Eq. 8 extended):**
```
F^(k+1) = (diag(q^(k)) + diag(λ))^{-1} [(W^(k) ⊙ Ã)F^(k) + diag(λ)F^(0)]
```
where λ = [λ_1,...,λ_N] is the per-node confidence vector.

Since `diag(q) + diag(λ)` is diagonal, inversion is trivially element-wise:
```
f_i^(k+1) = [agg_i + λ_i · f_i^(0)] / (q_i + λ_i)
```
**Computation cost is identical to RUNG_learnable_gamma.**

## Architecture

The MLP in RUNG aggregates in class space (output dim = C = num_classes),
so `F^(0) = MLP(X)` is directly the pre-aggregation logits. No
encoder/classifier split is needed:

```
X ──► MLP ──► F^(0)  [N, C]
               │
               ├──► softmax ──► conf_i ──► λ_i [N]   (new computation)
               │
               └──► Layer 0: QN-IRLS(F^(0), W^(0), lam^(0), λ)
                        └──► Layer 1: QN-IRLS(F^(1), W^(1), lam^(1), λ)
                                 └──► ...
                                          └──► F^(K) [N, C] ── final logits
```

Lambda is computed once from F^(0) and reused unchanged for all K layers,
because λ_i measures trust in F^(0), which does not change during aggregation.

## Files Created
- `model/rung_confidence_lambda.py` — model definition
- `train_eval_data/fit_confidence_lambda.py` — training function
- `analyze_confidence_lambda.py` — analysis and plotting utilities
- `docs/changes/008_confidence_lambda.md` — this file

## Files Modified
- `exp/config/get_model.py` — added RUNG_confidence_lambda case + import
- `clean.py` — added dispatch + CLI args (alpha_init, confidence_mode, etc.)
- `run_all.py` — added RUNG_confidence_lambda CLI forwarding args
- `experiments/run_ablation.py` — added 4 new experiment configs

## Three Confidence Modes

| Mode                | Formula                            | Hypothesis                           |
|---------------------|---------------------------------------------|--------------------------------------|
| `protect_uncertain` | λ_i ∝ (1 − conf_i + ε)^α           | Uncertain nodes need more protection |
| `protect_confident` | λ_i ∝ conf_i^α                     | Confident nodes resist corruption    |
| `symmetric`         | λ_i ∝ (4·conf_i·(1−conf_i))^α      | Extreme predictions need less skip   |

After computing raw λ_i, optional normalization preserves mean:
```
λ_i = λ_i × (λ_base / mean(λ_j))
```
This ensures RUNG_confidence_lambda and RUNG_learnable_gamma have the same
average skip-connection strength, isolating redistribution from magnitude.

## Learnable Alpha

Alpha controls sharpness of the confidence-to-lambda mapping:
- α = 1: linear mapping (small spread in λ_i)
- α > 1: amplifies differences (uncertain/confident nodes get more extreme λ_i)
- α ≈ 0: degenerates to uniform λ (equivalent to RUNG_learnable_gamma)

Parameterization ensures α > 0.5 always:
```
raw_alpha → α = softplus(raw_alpha) + 0.5
```
Initialised so α ≈ `alpha_init` at training start.

## Gradient Flow

Unlike learnable gamma (which has its own Parameters), lambda is
**computed** from the model's outputs — no new graph-level parameters:
```
loss ← F^(K) ← QN-IRLS(λ_i) ← softmax(F^(0)) ← MLP(X)
                                      ↑
                               gradient w.r.t. raw_alpha
```
The model is rewarded for producing confident F^(0) predictions when those
lead to lower final loss. This creates a beneficial self-supervised signal.

## How to Run

```bash
# Primary experiment: protect_uncertain mode on Cora
python run_all.py --datasets cora --models RUNG_confidence_lambda \
                  --confidence_mode protect_uncertain \
                  --normalize_lambda True \
                  --alpha_init 1.0 \
                  --warmup_epochs 50 \
                  --max_epoch 300

# All three confidence modes
python run_all.py --datasets cora citeseer \
                  --models RUNG_learnable_gamma RUNG_confidence_lambda

# Ablation experiments
python experiments/run_ablation.py --experiment confidence_mode_comparison
python experiments/run_ablation.py --experiment normalize_ablation
python experiments/run_ablation.py --experiment alpha_sensitivity
python experiments/run_ablation.py --experiment warmup_ablation

# Analysis plots (after training)
python analyze_confidence_lambda.py
```

## Diagnostic Signals During Training

The training output logs `α`, `λ_std`, and `λ-conf_corr` at each log interval.

| Signal | Healthy | Warning → Fix |
|--------|---------|----------------|
| `α` | Converges to 0.5–5.0 | `α → 0`: mechanism inactive → increase `alpha_lr_factor`, decrease `alpha_reg_strength` |
| `λ_std` | Non-zero after warmup | `λ_std ≈ 0`: redistribution not happening → same fix |
| `λ-conf_corr` | Negative for `protect_uncertain` | Near 0: confidences not yet informative → increase `warmup_epochs` |

## Results
[FILL IN AFTER RUNNING]

| Budget | RUNG | RUNG_new_SCAD | RUNG_learnable_gamma | RUNG_conf_λ (protect_unc) | RUNG_conf_λ (protect_conf) |
|--------|------|---------------|----------------------|---------------------------|---------------------------|
| 0      |      |               |                      |                           |                           |
| 0.05   |      |               |                      |                           |                           |
| 0.10   |      |               |                      |                           |                           |
| 0.20   |      |               |                      |                           |                           |
| 0.30   |      |               |                      |                           |                           |
| 0.40   |      |               |                      |                           |                           |

## Converged Alpha Value
[FILL IN: what alpha did the model learn? Expected range 0.5–3.0]

## Lambda Distribution (after training on clean graph)
[FILL IN: mean, std, range from `model.log_gamma_stats()`]

## Lambda-Confidence Correlation (clean graph)
[FILL IN: corr(λ, conf) from analyze / training logs]
