# 011 — Learnable Distance Metric (RUNG_learnable_distance)

## Date
2026-03-15

## Core Idea

Replace RUNG's fixed Euclidean distance (`y_ij = ||f_i/√d_i - f_j/√d_j||_2`) with a
configurable distance metric that can be:

- **Mode A: Cosine** (`1 - cosine_similarity`) — no new parameters, scale-invariant
- **Mode B: Projection** (learnable MLP projection) — ~few hundred parameters
- **Mode C: Mahalanobis/Bilinear** (learnable linear projection) — ~projecton_dim × out_dim parameters

### Why Better Distance Helps RUNG

RUNG's entire defense depends on computing edge "suspiciousness" scores `y_ij`:

- Small `y_ij` → edge is trusted, gets high weight in aggregation
- Large `y_ij` → edge is suspicious, gets pruned (or downweighted) by SCAD
- `y_ij > gamma` → edge is zeroed out completely

**The vulnerability**: An adversarial attacker can add edges between nodes with:
- Similar feature *norms* (Euclidean distance is small)
- But different feature *directions* (they're from different classes!)

Example: Two 64-dimensional embeddings with `||f_i|| ≈ ||f_j|| ≈ 10`:
- Euclidean: `||f_i - f_j||_2 ≈ 5` (small, not suspicious)
- Cosine: `angle(f_i, f_j) ≈ 90°` (very suspicious!)

Cosine distance ignores magnitude, focusing on *direction* — exactly what we need to catch
adversarial edges that hide by matching norms while changing direction.

### Why Percentile-Based Gamma Enables Easy Switching

When you replace Euclidean with cosine, the **scale of y_ij changes dramatically**:

```
Euclidean range:   0 to 5+ (depends on embedding norms, varies significantly)
Cosine range:      0 to 2  (bounded by definition, scale-invariant)
```

Traditional fixed-gamma approaches would require **manual re-tuning** when switching distances.

**But RUNG_percentile_gamma automatically adapts:**
```
gamma^(k) = quantile(y_ij_edges, percentile_q)
```

The gamma is always set to the `q`-th percentile of edge differences, **regardless of their scale**.
This means:
- Cosine at 75th percentile: ~1.0-1.5
- Euclidean at 75th percentile: ~3.0-5.0
- Both automatically tuned to prune the same fraction of edges

**Consequence**: No manual re-tuning needed when switching distance modes.
Same `percentile_q` and training procedure works for all distance metrics.

## Three Distance Modes

| Mode | Formula | Parameters | Scale | When to Use |
|------|---------|-----------|-------|------------|
| **cosine** | `1 - cos_sim(f̂_i, f̂_j)` | 0 | [0, 2] | **Always try first** — it's free |
| **projection** | `\|\|MLP(f̂_i) - MLP(f̂_j)\|\|_2` | ~hidden×proj_dim | [0, 2] | If cosine improves 1-2% |
| **bilinear** | `\|\|Wf̂_i - Wf̂_j\|\|_2` | proj_dim×out_dim | unbounded | If projection improves further |
| *Euclidean* (baseline) | `\|\|f̂_i - f̂_j\|\|_2` | 0 | [0, 5+] | Compare against this |

## Mathematical Details

### Mode A: Cosine Distance (NO NEW PARAMETERS)

```
f̂_i = F[i] / √d_i           (degree-normalized feature)
û_i = f̂_i / ||f̂_i||_2       (unit vector)

y_ij = 1 - û_i · û_j  ∈ [0, 2]
     = 1 - cosine_similarity(f̂_i, f̂_j)

Properties:
  - Scale-invariant: ||f̂_i|| and ||f̂_j|| don't matter, only direction
  - Symmetric: y_ij = y_ji
  - Always in [0, 2] (no scaling issues)
  - Zero gradient w.r.t. any parameter (but detection still works!)
```

**Why magnitude-invariance matters:**
If two nodes have the same direction but different norms:
- Euclidean sees them as "spread apart" (large distance)
- Cosine sees them as "aligned" (small distance, same class)

This is exactly what adversarial edges exploit — matching feature norms while changing direction.

### Mode B: Projection Distance (LEARNABLE MLP)

```
H_i = MLP(f̂_i)             (e.g., 64 → 32 → 16 features)
ĥ_i = H_i / ||H_i||_2      (normalize in projection space)

y_ij = ||ĥ_i - ĥ_j||_2  ∈ [0, 2]

MLP learns which dimensions best separate same-class from different-class pairs.
Parameters:
  - Linear(out_dim, hidden_dim//2): out_dim × 32  (e.g., 7 × 32 = 224)
  - Linear(hidden_dim//2, proj_dim): 32 × 16 = 512
  - Total: ~736 parameters (tiny)

Gradient flow: loss → y_ij → MLP → W → gradients for learning
```

### Mode C: Bilinear/Mahalanobis Distance (LEARNABLE PROJECTION)

```
H_i = W f̂_i                 (W ∈ ℝ^{proj_dim × out_dim}, no bias)

y_ij = ||H_i - H_j||_2

This is equivalent to Mahalanobis distance with metric M = W^⊤W.
More interpretable than full MLP (just linear basis change).

Parameters:
  - W: proj_dim × out_dim  (e.g., 16 × 7 = 112)
  - Total: 112 parameters (even tinier than Mode B)
```

## Model Lineage

```
RUNG (NeurIPS 2024)
  ├─ Base: Euclidean distance, fixed gamma
  │
  ├─ RUNG_new_SCAD, RUNG_new_L1, etc.
  │  └─ Penalty variations on fixed gamma
  │
  ├─ RUNG_learnable_gamma
  │  └─ Per-layer learnable gamma (K new parameters)
  │
  ├─ RUNG_percentile_gamma ← Use this as baseline
  │  └─ Per-layer adaptive gamma via percentile (0 new parameters)
  │
  └─ RUNG_learnable_distance (THIS FILE) ← NEW
     └─ Percentile gamma + configurable distance
        - Cosine: 0 params
        - Projection: ~700 params
        - Bilinear: ~100 params
```

## Key Differences from RUNG_percentile_gamma

| Aspect | RUNG_percentile_gamma | RUNG_learnable_distance |
|--------|----------------------|------------------------|
| Distance | Euclidean only | Configurable (cosine/projection/bilinear) |
| y_ij computation | `(f_i/√d_i - f_j/√d_j).norm()` | `distance_module(f_i/√d_i, f_j/√d_j)` |
| Distance parameters | 0 | 0 (cosine) or 100+ (projection/bilinear) |
| Gamma | Percentile-based | Percentile-based (unchanged) |
| Training | Single-group optimizer | Single or two-group optimizer |
| Typical y range | [0, 5+] | [0, 2] for cosine/projection |

## Files Created

1. `model/rung_learnable_distance.py`
   - `DistanceModule` class (handles 3 distance modes)
   - `RUNG_learnable_distance` class (like RUNG_percentile_gamma but with DistanceModule)

2. `train_eval_data/fit_learnable_distance.py`
   - `build_optimizer()` — single or two-group depending on distance mode
   - `fit_learnable_distance()` — training loop identical to fit_percentile_gamma
   - Automatically scales parameters and learning rates

3. `test_rung_learnable_distance.py`
   - 10 comprehensive checks covering all three modes
   - Validation of gradient flow, parameter counts, y ranges
   - End-to-end training verification

4. Updated files:
   - `clean.py`: Added model dispatch, argparse args for distance_mode and proj_dim
   - `exp/config/get_model.py`: Added RUNG_learnable_distance factory

## Usage

### Command-Line (via clean.py)

```bash
# Test cosine distance (no new parameters)
python clean.py --model RUNG_learnable_distance --data cora --distance_mode cosine

# Test projection distance with 32-dim projection
python clean.py --model RUNG_learnable_distance --data cora \
                 --distance_mode projection --proj_dim 32 --dist_lr_factor 0.5

# Test bilinear distance
python clean.py --model RUNG_learnable_distance --data cora \
                 --distance_mode bilinear --proj_dim 16

# Sweep percentile_q values  (inherited from RUNG_percentile_gamma)
for q in 0.50 0.75 0.90; do
  python clean.py --model RUNG_learnable_distance --data cora \
                   --distance_mode cosine --percentile_q $q
done
```

### Via run_all.py

```bash
# Compare all three distances on cora and citeseer
python run_all.py --datasets cora citeseer --models RUNG_learnable_distance \
                   --distance_mode cosine

# Longer training
python run_all.py --datasets cora citeseer --models RUNG_learnable_distance \
                   --distance_mode projection --max_epoch 500
```

### Programmatic (in Python)

```python
from model.rung_learnable_distance import RUNG_learnable_distance
from train_eval_data.fit_learnable_distance import fit_learnable_distance

# Create model
model = RUNG_learnable_distance(
    in_dim=dataset.num_features,
    out_dim=dataset.num_classes,
    hidden_dims=[64],
    distance_mode='cosine',  # or 'projection', 'bilinear'
    percentile_q=0.75,
    prop_step=10,
)

# Train
fit_learnable_distance(
    model, A, X, y, train_idx, val_idx,
    lr=0.05, dist_lr_factor=0.5, max_epoch=300
)
```

## Recommended Experiment Order

### Phase 1: Baseline Comparison
Establish that cosine distance works at all:

```bash
# Run on cora with default settings
python clean.py --model RUNG_percentile_gamma --data cora \
                 --percentile_q 0.75 --seed 0..4 → record accuracy

python clean.py --model RUNG_learnable_distance --data cora \
                 --distance_mode cosine --percentile_q 0.75 --seed 0..4
```

Expected result: cosine ≥ Euclidean baseline (or very close).

### Phase 2: Robustness Evaluation
If Phase 1 succeeds, test under adversarial attack:

```
python attack.py --model RUNG_learnable_distance --distance_mode cosine \
                  --data cora --budget 0.05 0.10 0.20
```

Expected result: cosine pruning is more effective → attack success lower.

### Phase 3: Further Optimization (only if Phase 2 shows >1% improvement)

```bash
# Try projection mode
python clean.py --model RUNG_learnable_distance --data cora \
                 --distance_mode projection --proj_dim 32

# Try different proj_dim values
for pd in 16 32 64; do
  python clean.py --model RUNG_learnable_distance --data cora \
                   --distance_mode projection --proj_dim $pd
done
```

## Theoretical Motivation

**Adversarial Edge Vulnerability in Euclidean Distance:**

Suppose nodes $i$ and $j$ are from different classes but attacker adds edge $(i,j)$.
After K layers of aggregation, RUNG computes:

$$y_{ij}^{(K)} = \left\| \frac{\mathbf{f}_i^{(K)}}{\sqrt{d_i}} - \frac{\mathbf{f}_j^{(K)}}{\sqrt{d_j}} \right\|_2$$

If $\|\mathbf{f}_i^{(K)}\| \approx \|\mathbf{f}_j^{(K)}\|$ (similar norms) but $\mathbf{f}_i^{(K)} \not\approx \mathbf{f}_j^{(K)}$ (different directions):
- Attacker can make $d_i \approx d_j$ by carefully choosing attack budget
- Then $y_{ij}^{(K)} \approx \| \mathbf{f}_i^{(K)} - \mathbf{f}_j^{(K)} \|_2 / \sqrt{d}$
- With modest attack budget, this can be made $< \gamma$ → edge not pruned → attack succeeds

**Why Cosine Distance is Robust:**

$$y_{ij}^{\text{cos}} = 1 - \frac{\mathbf{f}_i^{(K)} \cdot \mathbf{f}_j^{(K)}}{\|\mathbf{f}_i^{(K)}\| \cdot \|\mathbf{f}_j^{(K)}\|}$$

The norms $\|\mathbf{f}_i\|$ and $\|\mathbf{f}_j\|$ now **cancel out** (divided away).
Attacker cannot hide cross-class edges by matching norms — only direction matters.

If $\text{angle}(\mathbf{f}_i, \mathbf{f}_j) \approx 90°$ (different classes):
- Cosine distance ≈ 1.0 (very suspicious)
- Easy to prune, hard to attack

## Validation Checklist

- [ ] Test 1: Cosine mode forward pass and output shape
- [ ] Test 2: Cosine distance y values in [0, 2]
- [ ] Test 3: Cosine mode has 0 distance parameters
- [ ] Test 4: Projection mode has learnable parameters and gradient flow
- [ ] Test 5: Bilinear mode has learnable parameters
- [ ] Test 6: Percentile gamma computed automatically
- [ ] Test 7: Layerwise percentile q works correctly
- [ ] Test 8: Optimizer builder creates single/two-group as needed
- [ ] Test 9: End-to-end training step executes without errors
- [ ] Test 10: Model configuration and logging work
- [ ] Test 11: Run on cora dataset and report accuracy
- [ ] Test 12: Attack evaluation shows robustness improvement

## Expected Results (After Phase 1-2)

| Dataset | Model | Mode | Clean Acc | Attack@0.10 | Attack@0.20 |
|---------|-------|------|-----------|------------|------------|
| cora | RUNG_percentile (Euclidean) | - | 0.889 ± 0.008 | 0.445 | 0.123 |
| cora | RUNG_learnable_distance | cosine | 0.891 ± 0.010 | 0.520 | 0.180 |
| citeseer | RUNG_percentile (Euclidean) | - | 0.823 ± 0.012 | 0.412 | 0.145 |
| citeseer | RUNG_learnable_distance | cosine | 0.825 ± 0.013 | 0.485 | 0.195 |

*(Hypothetical; actual results depend on attack implementation and hyperparameters)*

## Notes for Future versions

1. **Modes B & C**: Only implement if Mode A shows >1% improvement on robustness
2. **Adaptive distance**: Could learn per-layer distance functions; adds complexity
3. **Distance ablation**: Separate study on which dimensions matter most (via attention)
4. **Heterophilic graphs**: Cosine might underperform if classes have similar embeddings by design
   - Future: conditional distance selection based on graph homophily

## References

- **Base model**: "Robust Graph Neural Networks via Unbiased Aggregation" (RUNG, NeurIPS 2024)
- **Cosine for robustness**: Cosine distance studied in adversarial ML for norm-invariance
- **Percentile thresholding**: Automatic threshold selection without parameter tuning (statistical learning)
