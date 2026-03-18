# 016 - RUNG_learnable_combined

## Date
2026-03-18

## Parent Model
RUNG_learnable_distance (cosine distance + percentile gamma)

## What Changes
Replaces percentile gamma with a learnable gamma designed for cosine distance range [0, 2].

## Why Previous Learnable Gamma (Euclidean) Failed
- Euclidean `y_ij` shrinks with layer depth (for example 0-8 early, 0-0.5 late).
- Fixed gamma initialization can leave deep layers with zero transition-zone edges.
- Gradient to gamma can vanish, causing unstable or seed-sensitive learning.

## Why Cosine Learnable Gamma Works
- Cosine `y_ij` is bounded in [0, 2] regardless of layer depth or seed.
- `gamma = sigmoid(raw) * 2.0` keeps gamma in (0, 2).
- SCAD transition region remains populated in cosine space.
- Gradient to gamma stays active, improving training stability.

## New Parameters vs RUNG_learnable_distance
| gamma_mode | New params |
|-----------|------------|
| per_layer | K scalars in one `raw_gamma` tensor |
| schedule  | 2 scalars: `raw_g0`, `raw_decay` |

## Files Created
- `model/rung_learnable_combined.py`
- `train_eval_data/fit_learnable_combined.py`
- `docs/changes/016_learnable_combined.md`

## Files Modified
- `exp/config/get_model.py` - registered `RUNG_learnable_combined`
- `clean.py` - added dispatch and `--gamma_mode`
- `attack.py` - added reconstruction case via saved params
- `run_all.py` - added `--gamma_mode` forwarding and summary output

## Key Diagnostic
After training, call `model.log_stats()` and check:
- Are learned gammas inside (0, 2)?
- Do per-layer gammas adapt across depth?
- If loss oscillates, reduce `gamma_lr_factor`.

## Results
Fill in after experiments.

| Dataset | Budget | RUNG | learnable_dist | learnable_combined (per_layer) | learnable_combined (schedule) |
|---------|--------|------|----------------|----------------------------------|---------------------------------|
| Cora    | 0%     |      |                |                                  |                                 |
| Cora    | 40%    |      |                |                                  |                                 |
| Citeseer| 40%    |      |                |                                  |                                 |
