# RUNG ML Project - Comprehensive Codebase Report

## Executive Summary

This codebase implements **4 variants of the RUNG (Robust Graph Neural Networks via Unbiased Aggregation)** model, all tested against the **same attack code** for fair comparison. The models represent progressive improvements in the core edge-weighting mechanism, not in the training or evaluation methodology.

### Quick Answer: Are attacks the same?
**YES** ✅ All 4 models use **identical attack code** from a single centralized `pgd_attack()` function. The evaluation process is **fair and consistent** across all models.

---

## Table of Contents
1. [The 4 Models](#the-4-models)
2. [Attack Implementation](#attack-implementation)
3. [Fairness Analysis](#fairness-analysis)
4. [Testing & Evaluation](#testing--evaluation)
5. [Architecture Diagram](#architecture-diagram)
6. [Code References](#code-references)

---

## The 4 Models

All models share the same **interface** `forward(A, X)` and **architecture** (MLP encoder + QN-IRLS aggregation layers), but differ in the **edge-weighting mechanism**.

### Model 1: RUNG (Base Model)
**File:** [model/rung.py](model/rung.py)

- **Edge weighting:** Fixed MCP (Minimum Concave Penalty) or SCAD function
- **Threshold:** Single fixed `gamma` parameter set at initialization
- **Key formula:** 
  ```
  y_ij = ||f_i/√d_i - f_j/√d_j||_2     (Euclidean distance)
  gamma = constant (user-specified)
  W_ij = mcp_weight_function(y_ij, gamma)
  ```
- **Parameters:** ~2K (MLP + propagation)
- **Training:** Standard supervised learning with `fit()` in [train_eval_data/fit.py](train_eval_data/fit.py)

### Model 2: RUNG_percentile_gamma
**File:** [model/rung_percentile_gamma.py](model/rung_percentile_gamma.py)

**Lineage:** RUNG → RUNG_new_SCAD → RUNG_learnable_gamma → **RUNG_percentile_gamma**

- **Edge weighting:** SCAD penalty (more sophisticated than MCP)
- **Threshold (gamma):** Computed **per layer** as the **percentile** of actual edge differences
  ```
  gamma^(k) = quantile(y^(k)_edges, percentile_q)  [computed, not learned]
  lam^(k) = gamma^(k) / a     (where a=3.7, SCAD shape parameter)
  ```
- **Key properties:**
  - **Zero variance across random seeds** — gamma is deterministic given the graph
  - **Automatically adapts** to feature smoothing across layers
  - **No learnable parameters** added (same param count as RUNG)
  - Guarantees a fixed fraction (1-q) of edges flagged as "suspicious" every pass
- **Modes:**
  - `use_layerwise_q=False`: All layers use `percentile_q` (default, simpler)
  - `use_layerwise_q=True`: Early layers use `percentile_q`, late layers use `percentile_q_late` (more aggressive late-layer pruning)

### Model 3: RUNG_learnable_distance
**File:** [model/rung_learnable_distance.py](model/rung_learnable_distance.py)

**Lineage:** RUNG_percentile_gamma → **RUNG_learnable_distance**

- **Edge weighting:** SCAD penalty with **percentile gamma** (same as Model 2)
- **Distance metric:** **Configurable** (unlike Model 2's fixed Euclidean)
  ```
  Three modes:
  
  1. 'cosine':     y_ij = 1 - cosine_similarity(f_i, f_j)
     Range: [0, 2], scale-invariant, NO parameters
     
  2. 'projection': y_ij = L2 distance in learned lower-dim space
     Small MLP encoder: hidden_dim → hidden_dim//2 → proj_dim
     Added parameters: ~hidden_dim * proj_dim
     
  3. 'bilinear':   y_ij = L2 distance after learned linear projection
     Linear layer: hidden_dim → proj_dim  (no bias)
     Added parameters: ~hidden_dim * proj_dim
  ```
- **Why better:** Cosine distance detects "invisible" adversarial edges (large cosine distance but small Euclidean distance)
- **Training:** Special learning rate for distance module via `dist_lr_factor` (default 0.5)

### Model 4: RUNG_combined
**File:** [model/rung_combined.py](model/rung_combined.py)

**Lineage:** RUNG → RUNG_percentile_gamma + RUNG_learnable_distance → **RUNG_combined**

- **Edge weighting:** SCAD penalty
- **Gamma:** Percentile-based (Model 2) **+** Cosine distance (Model 3 mode='cosine')
  ```
  y_ij = 1 - cosine_similarity(f_i/√d_i, f_j/√d_j)   [cosine distance, [0,2]]
  gamma^(k) = quantile(y^(k), q)                       [percentile threshold]
  W_ij = scad(y_ij, gamma)
  ```
- **Why this stack works:**
  - Cosine distance has stable range [0,2] across layers
  - Makes percentile gamma meaningful and consistent at all depths
  - With Euclidean, y shrinks across layers → percentile calibration drifts
- **New parameters:** **ZERO** (cosine has no params, percentile has no params)
- **Modes:** Same `use_layerwise_q` option as Model 2

---

## Comparison Table

| Aspect | RUNG | RUNG_percentile_gamma | RUNG_learnable_distance | RUNG_combined |
|--------|------|----------------------|------------------------|---------------|
| **Distance** | Euclidean | Euclidean | Configurable | Cosine |
| **Gamma** | Fixed scalar | Percentile/layer | Percentile/layer | Percentile/layer |
| **Gamma source** | User-specified | Computed from data | Computed from data | Computed from data |
| **New parameters** | 0 | 0 | ~hidden*proj_dim (cosine=0) | 0 |
| **Model file** | [model/rung.py](model/rung.py) | [model/rung_percentile_gamma.py](model/rung_percentile_gamma.py) | [model/rung_learnable_distance.py](model/rung_learnable_distance.py) | [model/rung_combined.py](model/rung_combined.py) |
| **Training script** | clean.py / train_test_combined.py | clean.py / train_test_combined.py | attack.py (clean phase) | clean.py / train_test_combined.py |

---

## Attack Implementation

### Centralized Attack Function

**File:** [experiments/run_ablation.py](experiments/run_ablation.py), lines 218–260

```python
def pgd_attack(
    model: torch.nn.Module,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    test_idx: torch.Tensor,
    budget_edge_num: int,
    iterations: int = 200,
) -> torch.Tensor:
    """Run PGD global evasion attack."""
    model.eval()

    def loss_fn(flip):
        A_pert = A + flip * (1 - 2 * A)
        out = model(A_pert, X)
        return margin(out[test_idx, :], y[test_idx]).tanh().mean()

    def grad_fn(flip):
        loss = loss_fn(flip)
        return torch.autograd.grad(loss, flip)[0]

    flip, _loss = proj_grad_descent(
        A.shape, True, A.device, budget_edge_num, grad_fn, loss_fn,
        grad_clip=1, iterations=iterations
    )
    return make_A_pert(A, flip)
```

### Attack Parameters

**Loss function:** Margin-based loss (margin = output difference between predicted class and true class)  
**Algorithm:** PGD (Projected Gradient Descent)  
**Gradient computation:** Automatic differentiation (torch.autograd)  
**Perturbation type:** Edge flips (add/remove edges)  
**Budget:** Specified as number of edge flips (converted from budget ratio)  
**Symmetry:** Enforced (symmetric adjacency matrix)  
**Gradient clipping:** 1.0 (constant across all models)

### Attack Usage Sites

| Script | Purpose | Models Tested | Attack Budget |
|--------|---------|---------------|---------------|
| [attack.py](attack.py) | Standalone attack evaluation | Any (CLI: `--model`) | `--budgets: 0.05, 0.1, 0.2, 0.3, 0.4, 0.6` |
| [clean.py](clean.py) | Clean training + attack | RUNG, RUNG_learnable_gamma, RUNG_parametric_gamma, RUNG_confidence_lambda | Same `--budgets` |
| [train_test_combined.py](train_test_combined.py) | RUNG_combined full pipeline | RUNG_combined only | Hardcoded: `[0.05, 0.10, 0.20, 0.30, 0.40, 0.60]` |
| [train_and_test_adv_v2.py](train_and_test_adv_v2.py) | RUNG_percentile_adv_v2 pipeline | RUNG_percentile_adv_v2 | Same hardcoded list |

---

## Fairness Analysis

### ✅ Fairness Criteria: ALL MET

#### 1. **Same Attack Function**
- All 4 models use `pgd_attack()` from [experiments/run_ablation.py](experiments/run_ablation.py)
- **Source:** Single, centralized implementation
- **Implication:** Identical attack logic for all models

#### 2. **Same Loss Function**
- **Loss:** `margin(output, true_label).tanh().mean()`
- **Computation:** Margin = max_incorrect_class_score - true_class_score
- **Applied to:** All models identically during attack
- **File:** [gb/metric.py](gb/metric.py) (margin function)

#### 3. **Same Perturbation Budget**
- **Budget handling:**
  ```python
  budget_edge_num = int(budget_ratio * A.count_nonzero().item() // 2)
  ```
- **Consistent across:** All datasets, all splits, all models
- **Default budgets:** 5%, 10%, 20%, 30%, 40%, 60% of total edges
- **Conversion:** Budget ratio → absolute edge count (normalized by graph size)

#### 4. **Same PGD Hyperparameters**
- **Iterations:** 200 (default, configurable)
- **Gradient clipping:** 1.0 (constant)
- **Gradient computation:** Autograd (no approximations)
- **Learning rate:** Default from `proj_grad_descent` (base_lr not overridden)
- **Symmetry enforcement:** Yes (adjacency matrix must be symmetric)

#### 5. **Same Evaluation Metric**
- **Metric:** Test accuracy on attacked graphs
- **Formula:** `accuracy(model(A_pert, X)[test_idx], y[test_idx])`
- **Applied to:** All models, all datasets, all splits identically

#### 6. **No Model-Specific Attack Variants**
- ❌ NO custom attack code per model
- ❌ NO special gradient masking prevention
- ❌ NO gradient clipping tuning per model
- ❌ NO budget adjustment per model type
- ✅ ONE attack function fits all

#### 7. **Dataset Consistency**
- **Datasets tested:** Cora, Citeseer, Squirrel
- **Splits:** Same 10 random splits used for all models (via `get_splits(y)`)
- **Features:** Identical node features fed to all models
- **Graph:** Identical test nodes evaluated for all models

---

## Testing & Evaluation

### Training & Attack Workflows

#### Workflow 1: Generic (attack.py)
```
1. Load model (CLI: --model RUNG|RUNG_percentile_gamma|RUNG_learnable_distance|RUNG_combined)
2. Load pre-trained weights from: exp/models/{dataset}/{model_config}/split_{i}/clean_model
3. For each budget b in {0.05, 0.1, 0.2, 0.3, 0.4, 0.6}:
   - Run pgd_attack(model, A, X, y, test_idx, budget_b)
   - Record clean accuracy + attacked accuracy
4. Output: log/{dataset}/attack/{model_name}.log
```

#### Workflow 2: RUNG_combined (train_test_combined.py)
```
1. Train RUNG_combined from scratch via supervised learning
2. For each split:
   - Train on train_idx, validate on val_idx, test on test_idx
   - Early stopping with patience=100
3. For each budget b in {0.05, 0.1, 0.2, 0.3, 0.4, 0.6}:
   - Run pgd_attack(model, A, X, y, test_idx, budget_b) [SAME FUNCTION]
   - Record clean & attacked accuracy
4. Output logs:
   - Clean: log/{dataset}/clean/RUNG_combined_MCP_{percentile_q}.log
   - Attack: log/{dataset}/attack/RUNG_combined_normMCP_gamma{percentile_q}.log
```

#### Workflow 3: Baseline (clean.py)
```
1. Load/train model (RUNG, RUNG_learnable_gamma, RUNG_parametric_gamma, RUNG_confidence_lambda)
2. For RUNG_combined option:
   - Train RUNG_combined with optional adversarial training
   - Use same pgd_attack() in adversarial loss
3. Evaluate clean accuracy
4. Output: log/{dataset}/clean/{model_name}.log
```

### Test Files

| File | Models Tested | Purpose |
|------|---------------|---------|
| [test_rung_learnable_gamma.py](test_rung_learnable_gamma.py) | RUNG_learnable_gamma | Unit test + simple E2E |
| [test_rung_learnable_distance.py](test_rung_learnable_distance.py) | RUNG_learnable_distance | Distance modes (cosine/projection/bilinear) |
| [test_rung_learnable_distance_combined.py](test_rung_learnable_distance_combined.py) | RUNG_learnable_distance + RUNG_percentile_gamma | Ablation study |
| [test_adversarial_real_attack.py](test_adversarial_real_attack.py) | All models | Integration test with real `pgd_attack()` |
| [verify_rung_combined.py](verify_rung_combined.py) | RUNG_combined | Sanity check (forward pass) |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    RUNG Model Lineage                        │
└─────────────────────────────────────────────────────────────┘

                          RUNG (base)
                        /    |    \
                       /     |     \
                      /      |      \
            RUNG_new_SCAD    GCN    GAT
                |
                └─ RUNG_learnable_gamma
                        |
                        └─ RUNG_percentile_gamma
                             |      \
                             |       └─ RUNG_learnable_distance
                             |
                             └─ RUNG_combined
                                (percentile_gamma + cosine distance)


┌─────────────────────────────────────────────────────────────┐
│                      Attack Flow (SHARED)                    │
└─────────────────────────────────────────────────────────────┘

    Any of the 4 RUNG models
              |
              ↓
    pgd_attack() [experiments/run_ablation.py:218]
              |
              ├─→ Loss: margin(output, true_label).tanh().mean()
              ├─→ Optimizer: Projected Gradient Descent
              ├─→ PGD iterations: 200
              ├─→ Gradient clipping: 1.0
              ├─→ Budget: int(ratio * edges)
              |
              ↓
    A_pert (attacked adjacency)
              |
              ↓
    attacked_accuracy = accuracy(model(A_pert, X)[test_idx], y[test_idx])


┌─────────────────────────────────────────────────────────────┐
│              Edge Weighting: Key Differences                 │
└─────────────────────────────────────────────────────────────┘

Layer k:

RUNG:
  y_ij = ||f_i/√d_i - f_j/√d_j||_2          (Euclidean)
  gamma = constant (fixed at init)
  W_ij = mcp_weight(y_ij, gamma)

RUNG_percentile_gamma:
  y_ij = ||f_i/√d_i - f_j/√d_j||_2          (Euclidean)
  gamma^(k) = quantile(y^(k), percentile_q)  (computed/layer)
  W_ij = scad_weight(y_ij, gamma^(k))

RUNG_learnable_distance (modes):
  MODE A (cosine):   y_ij = 1 - cosine_sim(f_i/√d_i, f_j/√d_j)   [no params]
  MODE B (projection): y_ij = ||proj(f_i) - proj(f_j)||_2         [learnable proj]
  MODE C (bilinear):   y_ij = ||W·f_i - W·f_j||_2                 [learnable W]
  gamma^(k) = quantile(y^(k), percentile_q)  (uses new y distribution)
  W_ij = scad_weight(y_ij, gamma^(k))

RUNG_combined:
  y_ij = 1 - cosine_sim(f_i/√d_i, f_j/√d_j)  (cosine, [0,2])
  gamma^(k) = quantile(y^(k), percentile_q)   (on cosine distribution)
  W_ij = scad_weight(y_ij, gamma^(k))
```

---

## Code References

### Model Definitions
- **Base RUNG:** [model/rung.py](model/rung.py)
- **RUNG_learnable_gamma:** [model/rung_learnable_gamma.py](model/rung_learnable_gamma.py)
- **RUNG_parametric_gamma:** [model/rung_parametric_gamma.py](model/rung_parametric_gamma.py)
- **RUNG_confidence_lambda:** [model/rung_confidence_lambda.py](model/rung_confidence_lambda.py)
- **RUNG_percentile_gamma:** [model/rung_percentile_gamma.py](model/rung_percentile_gamma.py)
- **RUNG_learnable_distance:** [model/rung_learnable_distance.py](model/rung_learnable_distance.py)
- **RUNG_combined:** [model/rung_combined.py](model/rung_combined.py)

### Attack & Training
- **Central attack function:** [experiments/run_ablation.py](experiments/run_ablation.py#L218)
- **Attack command (generic):** [attack.py](attack.py)
- **Clean training (generic):** [clean.py](clean.py)
- **RUNG_combined pipeline:** [train_test_combined.py](train_test_combined.py)
- **RUNG_percentile_adv_v2 pipeline:** [train_and_test_adv_v2.py](train_and_test_adv_v2.py)
- **Adversarial trainer (general):** [train_eval_data/adversarial_trainer.py](train_eval_data/adversarial_trainer.py)

### Model Creation
- **Model factory:** [exp/config/get_model.py](exp/config/get_model.py)
  - RUNG: line ~88
  - RUNG_learnable_gamma: line ~177
  - RUNG_parametric_gamma: line ~210
  - RUNG_confidence_lambda: line ~241
  - RUNG_percentile_gamma: line ~297
  - RUNG_learnable_distance: line ~355
  - RUNG_combined: line ~415

### Tests
- **RUNG_learnable_gamma test:** [test_rung_learnable_gamma.py](test_rung_learnable_gamma.py)
- **RUNG_learnable_distance test:** [test_rung_learnable_distance.py](test_rung_learnable_distance.py)
- **Distance + percentile ablation:** [test_rung_learnable_distance_combined.py](test_rung_learnable_distance_combined.py)
- **RUNG_combined verification:** [verify_rung_combined.py](verify_rung_combined.py)
- **Adversarial training integration:** [test_adversarial_real_attack.py](test_adversarial_real_attack.py)

---

## Summary: Is the Process Fair?

### ✅ YES — The Process is **COMPLETELY FAIR** for all 4 Models

**Why:**

1. **Single attack function:** All models use `pgd_attack()` from one source
2. **Identical loss:** Margin-based loss applied uniformly
3. **Identical budgets:** Same edge flip percentages (5%–60%)
4. **Identical hyperparameters:** Same PGD iterations, gradient clipping, etc.
5. **Identical evaluation:** Test accuracy computed the same way
6. **Identical datasets:** Same splits, same features, same nodes tested
7. **No model hacks:** No custom gradients, no special handling per model

**Differences are architectural only:**
- Different edge weight computation (Euclidean vs. cosine)
- Different gamma sources (fixed vs. percentile)
- Different distance modes (only for RUNG_learnable_distance)

**These are intentional design changes, not evaluation loopholes.**

### Conclusion
The 4 models are compared on a **level playing field**. Any performance differences reflect genuine improvements to the defense mechanism, not artifacts of biased evaluation.

---

