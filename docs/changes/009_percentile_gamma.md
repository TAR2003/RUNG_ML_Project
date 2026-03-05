# 009 — Percentile-Based Adaptive Gamma (RUNG_percentile_gamma)

## Date
2026-03-05

## Parent Model
`RUNG_learnable_gamma`

## New Model Name
`RUNG_percentile_gamma`

## Core Idea in One Sentence
Replace the learned `log_gamma` parameter with the q-th percentile of
the current edge difference distribution, computed automatically every
forward pass — no training of gamma needed at all.

---

## Why This Fixes Two Problems At Once

### Problem 1 — Fixed gamma fails at deep layers

At layer 10, features are smooth and edge differences are small.
Fixed `gamma=6.0` means nothing is ever pruned at layer 10.
The defense is disabled at the layers closest to the output.

Percentile gamma automatically shrinks with the distribution:
```
Layer  0: y_ij ranges 0–8.0  → gamma^(0) ≈ 6.0  (75th pct)
Layer  5: y_ij ranges 0–3.0  → gamma^(5) ≈ 2.3
Layer 10: y_ij ranges 0–0.5  → gamma^(9) ≈ 0.38
```
This means edge pruning remains active at every layer depth.

### Problem 2 — Learnable gamma has high variance across seeds

Gradient flows to gamma **only** through edges in the transition zone
(`gamma ≤ y < a·gamma`). If gamma initialises badly:
- All edges in region 1 (y < gamma): gradient = 0, gamma never moves
- All edges in region 3 (y ≥ a·gamma): gradient = 0, gamma never moves

This causes the instability seen in `RUNG_learnable_gamma`:
~30% variance between seeds depending on initialisation.

Percentile gamma **cannot get stuck** — it always positions itself
relative to the current distribution. Every forward pass uses the same
quantile logic regardless of seed.

---

## What Changed vs RUNG_learnable_gamma

| Aspect              | RUNG_learnable_gamma         | RUNG_percentile_gamma             |
|---------------------|------------------------------|-----------------------------------|
| Gamma source        | `exp(log_lam)` parameter     | `quantile(y_edges, q)` every fwd  |
| Parameters          | +K scalars (log_lam)         | 0 new parameters                  |
| Gradient to gamma   | Yes (backprop)               | No (not a parameter)              |
| Variance from gamma | High (init-dependent)        | Zero (deterministic)              |
| Main hyperparameter | `gamma_init` + `gamma_lr`    | `percentile_q`                    |
| Tuning difficulty   | High (LR, init, reg)         | Low (single q value)              |
| Depth adaptation    | Only if gradient flows       | Always (by construction)          |

---

## The Percentile Formula

At layer k, with current node features F^(k):

```
Step 1: Compute normalised features
    f̃_i = f_i / √d_i     for all nodes i

Step 2: Compute edge differences
    y_ij = || f̃_i - f̃_j ||_2     for all edges (i,j)

Step 3: Compute percentile gamma (only over off-diagonal edges)
    gamma^(k) = quantile({y_ij : (i,j) ∈ E, i≠j}, q)
    lam^(k)   = gamma^(k) / a

Step 4: Use lam^(k) in SCAD weight as usual
    W_ij = scad_weight_differentiable(y_ij, lam^(k))
```

---

## What `q` Controls

```
q = 0.50 → gamma = median of y  → prune top 50% → very aggressive
q = 0.75 → gamma = 75th pct     → prune top 25% → moderate defense
q = 0.90 → gamma = 90th pct     → prune top 10% → light defense
q = 0.95 → gamma = 95th pct     → prune top 5%  → very light
q = 1.00 → gamma = max(y)       → nothing pruned → no defense
```

Adversarial edges have **larger** y values (attacker adds edges between
dissimilar nodes). So the top (1-q) fraction of edges by difference is
enriched for adversarial edges, making this theoretically sound.

---

## Files Created

| File | Purpose |
|------|---------|
| `model/rung_percentile_gamma.py` | Model class |
| `train_eval_data/fit_percentile_gamma.py` | Training loop (simpler than learnable_gamma) |
| `experiments/search_percentile_q.py` | Grid search over q values |
| `analyze_percentile_gamma.py` | Gamma profile and variance analysis |
| `docs/changes/009_percentile_gamma.md` | This file |

## Files Modified

| File | Change |
|------|--------|
| `exp/config/get_model.py` | Added `RUNG_percentile_gamma` import + factory case |
| `clean.py` | Added import, argparse args, model dispatch |
| `run_all.py` | Added `--percentile_q`, `--use_layerwise_q`, `--percentile_q_late` args and forwarding |
| `attack.py` | Added percentile args and passed to model_params for reconstruction |

---

## How to Run

```bash
# Step 1: Find best q value on cora (run this first)
python experiments/search_percentile_q.py --dataset cora

# Step 2: Train with fixed best q
python run_all.py --datasets cora citeseer --models RUNG_percentile_gamma \
                  --percentile_q 0.75

# Step 3: Try layerwise q (more aggressive pruning in later layers)
python run_all.py --datasets cora --models RUNG_percentile_gamma \
                  --use_layerwise_q True \
                  --percentile_q 0.85 \
                  --percentile_q_late 0.65

# Step 4: Analyse gamma profiles after training
python analyze_percentile_gamma.py determinism --dataset cora --percentile_q 0.75

# Step 5: Plot q sensitivity from search results
python analyze_percentile_gamma.py q_sensitivity \
    --csv results/comparison/q_search_cora_<timestamp>.csv \
    --budget 0.40
```

---

## Layerwise q Option

When `--use_layerwise_q True`:
```
Layers 0 .. K//2 - 1:  use percentile_q       (e.g. 0.85 — lighter early)
Layers K//2 .. K - 1:  use percentile_q_late   (e.g. 0.65 — heavier late)
```

Motivation: features are more heterogeneous in early layers (large legitimate
differences), while late layers have smoothed features where adversarial edges
stand out more clearly. Stronger late-layer pruning is more targeted.

Start with `use_layerwise_q=False` (simpler). Switch to `True` only if the
single-q version shows systematically worse late-layer pruning.

---

## Expected Results

Vs `RUNG_learnable_gamma`:
- **Std across seeds: much lower** (key claim — gamma determinism)
- Mean attacked accuracy: similar or better
- Clean accuracy: similar

Key result to report:
> "RUNG_learnable_gamma achieves X% attacked accuracy with std=Y%.
>  RUNG_percentile_gamma achieves X'% with std=Y'%, where Y' << Y,
>  confirming that percentile adaptation eliminates the instability
>  of gradient-based gamma learning."

---

## Diagnostic: Do Gammas Decrease With Depth?

After training, call `model.log_gamma_stats()` (runs automatically at end of
`fit_percentile_gamma`). Expected output:

```
======================================================
     RUNG_percentile_gamma — Gamma Values (last fwd)
======================================================
 Layer     gamma       lam    q used
------------------------------------------------------
     0     4.2103    1.1380    0.75
     1     3.8821    1.0492    0.75
     2     3.1204    0.8433    0.75
     ...
     9     0.9234    0.2496    0.75
======================================================
```

If gammas do **not** decrease: check `lam_hat`. Very low `lam_hat` means
the skip connection dominates and features do not smooth across layers.

---

## Best q Values (fill in after running search)

| Dataset    | Best q | Clean Acc | Attacked 40% |
|------------|--------|-----------|--------------|
| Cora       |        |           |              |
| Citeseer   |        |           |              |

---

## Notes

**Note 1 — Determinism is your most important result:**
The gamma from `RUNG_percentile_gamma` is identical across seeds given the
same graph. Run `analyze_percentile_gamma.py determinism` to confirm. Report
this explicitly as evidence for the stability claim.

**Note 2 — Start q search before anything else:**
The best q value varies by dataset. Running `search_percentile_q.py` first
saves days of running the wrong configuration.

**Note 3 — Self-loops excluded from percentile:**
The percentile is computed over off-diagonal edge differences only. Self-loop
differences are exactly 0 and would corrupt the distribution at low q.
The implementation uses `A_bool & ~eye_bool` as the edge mask.

**Note 4 — No separate gamma LR needed:**
`fit_percentile_gamma.py` uses a single Adam group. There is no `gamma_lr_factor`
to tune. This is simpler than `RUNG_learnable_gamma`.
