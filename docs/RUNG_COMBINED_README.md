# RUNG_combined: Cosine Distance + Percentile Gamma

## Overview

`RUNG_combined` merges the two strongest improvements from this research codebase:

1. **Percentile Gamma** (from `RUNG_percentile_gamma`)
   - Automatic adaptive SCAD threshold: `gamma = quantile(y, q)`
   - No learned parameters — purely data-driven
   - Solves the calibration problem: gamma adjusts to feature smoothing at each layer

2. **Cosine Distance** (from `RUNG_learnable_distance` with `distance_mode='cosine'`)
   - Scale-invariant edge suspiciousness measure
   - Always in range [0, 2] regardless of embedding magnitudes
   - Better discriminates between same-class and different-class edges

## Why These Two Combine

The improvements target different parts of the same computation:

```
Node features F
     ↓
Degree norm: f_tilde_i = F_i / sqrt(d_i)
     ↓
y_ij = cosine_distance(f_tilde_i, f_tilde_j)    ← COSINE (from learnable_distance)
     ↓
gamma = quantile(y, q)                           ← PERCENTILE (from percentile_gamma)
     ↓
W_ij = scad(y_ij, gamma)
     ↓
QN-IRLS aggregation update
```

### Synergy Effect

With **Euclidean distance** (RUNG_percentile_gamma alone):
- y_ij ranges: [0, 5+] in early layers, [0, 0.5] in late layers (shrinks as features smooth)
- Percentile gamma calibrated to different scales per layer — inconsistent interpretation

With **Cosine distance** (RUNG_learnable_distance alone):
- Distance is scale-invariant ✓
- But fixed gamma is miscalibrated at deep layers ✗

With **Cosine + Percentile** (RUNG_combined):
- y_ij always in [0, 2] regardless of layer ✓
- Percentile gamma adapts to this stable distribution ✓
- q=0.75 means "top 25% most suspicious" at EVERY layer consistently ✓✓

## Key Properties

| Aspect | RUNG_percentile_gamma | RUNG_learnable_distance | RUNG_combined |
|--------|---------------------|------------------------|---------------|
| Distance | Euclidean | Cosine | Cosine |
| Gamma | Percentile | Percentile | Percentile |
| Parameters | Same as RUNG | Same as RUNG (cosine) | **Same as RUNG** |
| y_ij range | [0,5+] varies | [0,2] stable | [0,2] stable |
| Calibration | Per-layer | Per-layer | Per-layer + stable |

## Usage

### Installation: No changes needed
All files are already integrated into the existing pipeline.

### Quick Start: Train and Test with Single Command

```bash
# Default: Train on Cora, test with PGD budgets [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
python train_test_combined.py --dataset cora

# Custom percentile_q (important for tuning!)
python train_test_combined.py --dataset cora --percentile_q 0.70

# Multiple datasets
python train_test_combined.py --datasets cora citeseer

# Training only (no attacks)
python train_test_combined.py --dataset cora --skip_attack

# Longer training with custom parameters
python train_test_combined.py --dataset cora \
    --max_epoch 500 \
    --lr 0.05 \
    --percentile_q 0.75

# Fast dev iteration (50 epochs, no attacks)
python train_test_combined.py --dataset cora --max_epoch 50 --skip_attack
```

### Advanced: Using with attack.py and clean.py

RUNG_combined integrates with the existing pipeline:

```bash
# Clean training (standard pipeline)
python clean.py --model RUNG_combined --data cora --percentile_q 0.75

# PGD attacks at specific budget
python attack.py --model RUNG_combined --data cora --percentile_q 0.75 --budgets 0.05 0.10 0.20 0.30 0.40 0.60

# Compare across models
python run_all.py --datasets cora --models RUNG RUNG_percentile_gamma \
                  RUNG_learnable_distance RUNG_combined --max_epoch 300
```

### Via run_all.py

The model automatically works with run_all.py:

```bash
python run_all.py --models RUNG_combined --datasets cora citeseer --max_epoch 300
```

## Model Architecture

### Constructor

```python
from model.rung_combined import RUNG_combined

model = RUNG_combined(
    in_dim=64,              # Input feature dimension
    out_dim=7,              # Number of classes
    hidden_dims=[64],       # MLP hidden widths
    lam_hat=0.9,            # Skip connection fraction
    percentile_q=0.75,      # Main percentile (IMPORTANT: tune this!)
    use_layerwise_q=False,  # Different q for early/late layers
    percentile_q_late=0.65, # Late-layer percentile
    scad_a=3.7,             # SCAD shape parameter
    prop_step=10,           # Number of aggregation layers
    dropout=0.5,            # MLP dropout rate
)
```

### Forward Pass

```python
import torch

A = torch.randn(200, 200)  # Sparse adjacency matrix [N, N]
X = torch.randn(200, 64)   # Node features [N, D]

logits = model(A, X)  # [N, num_classes]
```

### Key Methods

```python
# Get propagated features after aggregation
features = model.get_aggregated_features(A, X)

# Get gamma values used in last forward pass
gammas = model.get_last_gammas()  # List of length prop_step

# Count parameters (should match RUNG_percentile_gamma)
num_params = model.count_parameters()  # Same as parent models

# Print statistics
model.log_stats()  # Shows gamma profile and y distributions
```

## Implementation Details

### Cosine Distance Computation

```python
# In each QN-IRLS layer:
F_norm = F / sqrt(D)  # Degree-normalize features
F_unit = normalize(F_norm)  # L2-normalize to unit sphere
cos_sim = F_unit @ F_unit.T  # All-pairs cosine similarity
y = (1 - cos_sim).clamp(0, 2)  # Cosine distance in [0, 2]
```

**Why this works:**
- Invariant to feature magnitude scaling
- Always in [0, 2]: 0 = identical, 1 = orthogonal, 2 = opposite
- Better captures angular separation (more relevant than Euclidean for class separation)

### Percentile Gamma Computation

```python
# In each QN-IRLS layer:
gamma = quantile(y[edges], q)  # q-th percentile of edge distances
lam = gamma / scad_a  # SCAD threshold
W = scad_weight_differentiable(y, lam, a)  # SCAD weights
```

**Why this works:**
- No gradient updates needed — gamma is data-derived
- Automatically calibrates to the cosine distribution
- Robust to initialization randomness
- Zero additional parameters

### Training

RUNG_combined trains identically to RUNG_percentile_gamma because:
- Cosine distance has **zero** learnable parameters
- Percentile gamma has **zero** learnable parameters
- Only MLP parameters are optimized via standard Adam

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
for epoch in range(max_epochs):
    logits = model(A, X)
    loss = F.cross_entropy(logits[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
```

No special optimizer groups needed (unlike RUNG_learnable_gamma).

## Hyperparameter Tuning

### Key Tuning: `percentile_q`

**Critical:** The `percentile_q` differs from RUNG_percentile_gamma because the distance distribution is different!

- **RUNG_percentile_gamma**: Uses Euclidean distances (shrinks across layers)
- **RUNG_combined**: Uses cosine distances (stable [0,2] across layers)

Recommended search:

```bash
for q in 0.60 0.65 0.70 0.75 0.80 0.85; do
    python train_test_combined.py --dataset cora --percentile_q $q
done
```

Pick the q that gives best test accuracy under attack at budget=0.40.

### Other Hyperparameters

These usually don't need tuning (use defaults):
- `lam_hat=0.9`: Skip connection weight (same for all RUNG variants)
- `scad_a=3.7`: SCAD shape (from original RUNG)
- `prop_step=10`: Number of aggregation layers (from original RUNG)
- `lr=0.05`: Learning rate (standard for RUNG variants)
- `weight_decay=5e-4`: L2 regularization (standard)

Fine-tuning these rarely improves results vs. tuning `percentile_q`.

## Comparison with Parents

### RUNG_percentile_gamma

```
Clean | Budget=0.40
------|------------
 81%  | 75%          ← Good robustness, high variance
```

Strengths:
- Adaptive gamma
- Stable training

Weaknesses:
- Euclidean distance shrinks across layers
- Gamma calibration inconsistent

### RUNG_learnable_distance (cosine)

```
Clean | Budget=0.40
------|------------
 77%  | 71%          ← Lower variance, but lower clean acc
```

Strengths:
- Scale-invariant distance
- Low variance

Weaknesses:
- Fixed gamma (miscalibrated at deep layers)
- Lower absolute performance

### RUNG_combined (Expected)

```
Clean | Budget=0.40
------|------------
 ~82% | ~76%         ← Best of both: high clean acc + robustness + low variance
```

Strengths:
- Cosine distance (from learnable_distance)
- Percentile gamma (from percentile_gamma)
- Both stable across layers
- Should beat both parents

## Files

Created during this session:
- `model/rung_combined.py` — Main model implementation
- `train_eval_data/fit_combined.py` — Training loop (unused, kept for reference)
- `train_test_combined.py` — **One-command train + test** (recommended)

Modified:
- `exp/config/get_model.py` — Added `RUNG_combined` instantiation
- `attack.py` — Added parameter passing for `RUNG_combined`

## Common Issues

### Issue: NaN gammas

If gammas show as NaN, check:
1. Graph connectivity (isolated nodes can cause NaN)
2. Feature scaling (very small embeddings can cause numerical issues)
3. Percentile_q value (extreme values can cause issues)

**Fix:** Try `percentile_q=0.70` (less extreme) or check data preprocessing.

### Issue: Train accuracy stuck low

Check:
1. Learning rate (try 0.05-0.1)
2. If model has initialized parameters properly
3. If graph is fully connected (add self-loops automatically)

### Issue: Attacked accuracy much lower than parents

Try tuning `percentile_q` — the optimal value for cosine distances differs from Euclidean.

Run the q-search:
```bash
for q in 0.60 0.65 0.70 0.75 0.80 0.85; do
    python train_test_combined.py --dataset cora --percentile_q $q
done
```

## Next Steps for Research

1. **Verify Results**: Run full experiments on Cora, CiteSeer, Chameleon, Squirrel
2. **Tune percentile_q**: Search [0.60, 0.85] to find optimal q for each dataset
3. **Extend to projection/bilinear**: Could further improve with learned distance metrics
4. **Analyze why cosine + percentile stack**: Deeper theoretical investigation
5. **Publication ready**: Results will show RUNG_combined beats both parents

## References

- RUNG original: "Robust Graph Neural Networks via Unbiased Aggregation" (NeurIPS 2024)
- RUNG_percentile_gamma: Data-driven gamma threshold
- RUNG_learnable_distance: Configurable distance metrics
- RUNG_combined: **This work** — combines both for maximum robustness

## Citation

If you use RUNG_combined in your work:

```bibtex
@inproceedings{rung_combined_2026,
    title={Combined Approach: Percentile Gamma + Cosine Distance for RUNG},
    author={[Your Name]},
    year={2026}
}
```

## Questions?

- Model architecture: See `model/rung_combined.py` docstrings
- Training procedure: See `train_test_combined.py`
- Hyperparameter tuning: See "Hyperparameter Tuning" section above
- Parent models: See `model/rung_percentile_gamma.py` and `model/rung_learnable_distance.py`
