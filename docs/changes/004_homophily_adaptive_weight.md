# 004 — Homophily-Aware Adaptive Edge Weight

## Date
2026-02-27

## Motivation
RUNG's MCP penalty assumes large feature differences signal adversarial edges.
On heterophilic graphs (h < 0.3) this assumption is systematically WRONG —
different-class nodes are legitimately connected and have large feature
differences. The adaptive weight uses soft class predictions to distinguish
legitimate heterophilic edges from adversarial ones.

## Mathematical Formulation

**Heterophily score per edge:**
$$h_{ij} = 1 - \sum_c p_{ic} \cdot p_{jc}$$
where $p_{ic}$ is the predicted class probability for node $i$.
- $h_{ij} = 0$ → nodes predicted same class (homophilic edge)
- $h_{ij} = 1$ → nodes predicted fully different classes (heterophilic edge)

**Homophilic weight (standard MCP):**
$$W^{\text{homo}}_{ij} = \max\left(0, \frac{1}{2y} - \frac{1}{2\gamma}\right)$$

**Heterophilic weight (inverted MCP):**
$$W^{\text{hetero}}_{ij} = \max\left(0, \frac{1}{2\gamma} - \frac{1}{2y}\right)$$

**Adaptive weight:**
$$W_{ij} = (1 - h_{ij}) \cdot W^{\text{homo}}_{ij} + h_{ij} \cdot W^{\text{hetero}}_{ij}$$

**Interpretation:**
- On a homophilic edge (h≈0): uses standard MCP (suppress adversarial different-feature edges)
- On a heterophilic edge (h≈1): uses inverted weight (preserve cross-class edges with large features)

## Implementation Notes

### Dense matrix adaptation
The codebase uses dense [N, N] adjacency matrices (not PyG edge_index).
`homophily_adaptive` computes the full [N, N] dot product matrix:
```python
dot_product = soft_labels @ soft_labels.t()   # [N, C] × [C, N] → [N, N]
h = 1.0 - dot_product.clamp(0.0, 1.0)        # [N, N] heterophily
```

### Soft labels from current iteration
At each QN-IRLS propagation step, soft labels are computed from the
CURRENT feature matrix F (not the initial F⁰):
```python
soft_labels = torch.softmax(F.detach(), dim=-1)  # [N, C]
```
Using `.detach()` prevents gradients flowing through the soft labels,
which would create a second-order gradient path.

### First-iteration stability
At iteration 0, F = F⁰ (MLP output, not yet propagated). On random init,
F⁰ is noisy and soft labels are nearly uniform (h ≈ 1/C for all edges).
This means the adaptive weight averages W_homo and W_hetero roughly equally,
which is a safe initialisation that converges as propagation progresses.

## What Changed

### Modified Files
- **`model/penalty.py`**: Added `PenaltyFunction.homophily_adaptive(y, gamma, soft_labels)`
  at lines ~150–197. Works with [N, N] dense matrices.
- **`model/rung.py`**: Extended `forward()` with adaptive branch:
  ```python
  if self.penalty == 'adaptive':
      soft_labels = torch.softmax(F.detach(), dim=-1)
      W = PenaltyFunction.homophily_adaptive(y, self.gamma, soft_labels)
  else:
      W = self.w(y)
  ```
- **`exp/config/get_model.py`**: Added `'ADAPTIVE'` case in `get_model_default()`:
  passes `penalty='adaptive'` to RUNG constructor (w_func is MCP as fallback).
- **`clean.py`** / **`attack.py`**: `--penalty adaptive` maps to `--norm ADAPTIVE`.

## How to Use
```bash
# Recommended for heterophilic datasets:
python clean.py --model='RUNG' --penalty=adaptive --gamma=3.0 --data='chameleon'
python clean.py --model='RUNG' --penalty=adaptive --gamma=2.0 --data='texas'

# Then attack:
python attack.py --model='RUNG' --penalty=adaptive --gamma=3.0 --data='chameleon'
```

## Limitations and Future Work
1. **Circular dependency risk**: Soft labels depend on F, which depends on
   the weights, which depend on soft labels. Convergence is not theoretically
   guaranteed for adaptive mode (unlike QN-IRLS with fixed weights). In practice,
   the IRLS outer loop tends to converge empirically.
2. **First-layer warm-up**: Consider running the first 2 iterations with MCP
   before switching to adaptive mode for more stable initialisation.
3. **Computational cost**: Computing `soft_labels @ soft_labels.t()` is O(N²C),
   same as the existing `pairwise_squared_euclidean` call — no extra asymptotic cost.
