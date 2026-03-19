# Enhanced run_all.py - 4-Model Comparison Guide

## Quick Start: Run All 4 Models in One Command

```bash
python run_all.py --datasets cora citeseer --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined
```

This single command will:
1. ✅ Train all 4 models on both datasets (Cora & Citeseer)
2. ✅ Attack each model with extended budgets: 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
3. ✅ Generate comparable logs in `log/<dataset>/clean/` and `log/<dataset>/attack/`
4. ✅ Display summary table at the end

**Expected output:**
```
════════════════════════════════════════════════════════
  Train (clean)  —  3 model(s) × 2 dataset(s)
════════════════════════════════════════════════════════
>>> [Train (clean)]  dataset=cora  model=RUNG  norm=MCP  gamma=6.0
... [training output] ...
<<< DONE  (45.3s)

>>> [Train (clean)]  dataset=cora  model=RUNG_percentile_gamma  ...
... [training output] ...
<<< DONE  (52.1s)

... [more models] ...

════════════════════════════════════════════════════════
  Train + Attack (unified)  —  1 model(s) × 2 dataset(s)
════════════════════════════════════════════════════════
>>> [Train + Attack]  dataset=cora  model=RUNG_combined  percentile_q=0.75
... [training + attack output] ...
<<< DONE  (78.5s)

... [citeseer] ...

════════════════════════════════════════════════════════
  Summary
════════════════════════════════════════════════════════
  Phase                         Model               Dataset        Script                 Status    Time
  ──────────────────────────────────────────────────────────────────────────────────────────────────────
  Train (clean)                 RUNG                cora           clean.py               OK        45.3s
  Train (clean)                 RUNG                citeseer       clean.py               OK        44.2s
  PGD attack                    RUNG                cora           attack.py              OK        123.4s
  PGD attack                    RUNG                citeseer       attack.py              OK        118.7s
  Train (clean)                 RUNG_percentile_gamma  cora        clean.py               OK        52.1s
  ...
  Train + Attack (unified)      RUNG_combined       cora           train_test_combined.py OK        78.5s
  Train + Attack (unified)      RUNG_combined       citeseer       train_test_combined.py OK        76.2s

  Total: 12/12 passed  |  wall-time: 1243.5s

  Model-specific configuration:
    RUNG & RUNG_learnable_gamma:  MCP/SCAD penalty, gamma=6.0
    RUNG_percentile_gamma:        Percentile-based gamma, percentile_q=0.75
    RUNG_learnable_distance:      Distance=cosine, percentile_q=0.75
    RUNG_combined:                Cosine distance + percentile_q=0.75
    Attack budgets:               [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

  ✅ All jobs passed!
  Run  python plot_logs.py  to visualise and compare all 4 models.
```

---

## Command Variations

### 1. Run on Single Dataset
```bash
python run_all.py --datasets cora --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined
```

### 2. Run with Custom Attack Budgets (Extended)
```bash
python run_all.py --datasets cora citeseer \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined \
  --budgets 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
```

### 3. Run Specific Models
```bash
# Only RUNG and RUNG_combined
python run_all.py --datasets cora citeseer --models RUNG RUNG_combined

# Only learnable variants
python run_all.py --datasets cora citeseer \
  --models RUNG_percentile_gamma RUNG_learnable_distance
```

### 4. Skip Attack (Training Only)
```bash
python run_all.py --datasets cora citeseer \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined \
  --skip_attack
```

### 5. Skip Training (Attack Only, Reuse Existing Models)
```bash
python run_all.py --datasets cora citeseer \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined \
  --skip_clean
```

### 6. Custom Hyperparameters
```bash
python run_all.py --datasets cora citeseer \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined \
  --gamma 8.0 \
  --max_epoch 500 \
  --percentile_q 0.80 \
  --distance_mode cosine
```

### 7. Test Extended Percentile Values
```bash
python run_all.py --datasets cora citeseer \
  --models RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined \
  --percentile_q 0.70
```

---

## The 4 Models Explained

### Model 1: RUNG (Baseline)
- **Distance:** Euclidean (fixed)
- **Gamma:** User-specified constant (default: 6.0)
- **Script:** `clean.py` → `attack.py`
- **New parameters:** None

```bash
python run_all.py --models RUNG --gamma 6.0
```

### Model 2: RUNG_percentile_gamma
- **Distance:** Euclidean (fixed)
- **Gamma:** Percentile-based, computed per layer (default: 0.75)
- **Script:** `clean.py` → `attack.py`
- **New parameters:** None

```bash
python run_all.py --models RUNG_percentile_gamma --percentile_q 0.75
```

### Model 3: RUNG_learnable_distance
- **Distance:** Configurable (cosine/projection/bilinear, default: cosine)
- **Gamma:** Percentile-based (default: 0.75)
- **Script:** `clean.py` → `attack.py`
- **New parameters:** `--distance_mode`, `--proj_dim`, `--dist_lr_factor`

```bash
python run_all.py --models RUNG_learnable_distance \
  --distance_mode cosine \
  --percentile_q 0.75
```

### Model 4: RUNG_combined
- **Distance:** Cosine (scale-invariant, best)
- **Gamma:** Percentile-based on cosine distribution (default: 0.75)
- **Script:** `train_test_combined.py` (unified train + attack)
- **New parameters:** None

```bash
python run_all.py --models RUNG_combined --percentile_q 0.75
```

---

## Important Parameters

### Extended Attack Budgets
```bash
--budgets 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
```
- Evaluates robustness at 11 different perturbation levels
- Default if not specified (recommended for comprehensive comparison)
- Each point represents 5%, 10%, ..., 100% of total edges possible to flip

### Percentile Gamma (Shared by Models 2-4)
```bash
--percentile_q 0.75           # Main percentile (default)
--use_layerwise_q False       # Use same q for all layers (default: True for best results)
--percentile_q_late 0.65      # Different q for late layers (only if use_layerwise_q=True)
```

### Distance Mode (RUNG_learnable_distance only)
```bash
--distance_mode cosine        # Scale-invariant, no learnable params (RECOMMENDED)
--distance_mode projection    # Learnable MLP projection
--distance_mode bilinear      # Learnable linear projection
```

---

## Log Files Generated

After running, check:
```
log/cora/clean/RUNG_MCP_6.0.log
log/cora/clean/RUNG_percentile_gamma_MCP_6.0.log
log/cora/clean/RUNG_learnable_distance_MCP_6.0.log
log/cora/clean/RUNG_combined_MCP_6.0.log

log/cora/attack/RUNG_normMCP_gamma6.0.log
log/cora/attack/RUNG_percentile_gamma_normMCP_gamma6.0.log
log/cora/attack/RUNG_learnable_distance_normMCP_gamma6.0.log
log/cora/attack/RUNG_combined_normMCP_gamma6.0.log
```

---

## Visualize Results

After all runs complete:
```bash
python plot_logs.py
```

This generates comparison plots showing:
- **Clean accuracy** of each model
- **Robustness curves** (accuracy vs attack budget)
- **Model comparison** across datasets

---

## Fairness of Comparison

✅ **All 4 models use identical attack code**
- Same `pgd_attack()` function from `experiments/run_ablation.py`
- Same loss function, same PGD parameters
- Same budgets, same evaluation metrics
- Fair comparison guaranteed

✅ **Architectural differences only:**
- RUNG: Fixed gamma
- RUNG_percentile_gamma: Data-driven per-layer gamma
- RUNG_learnable_distance: Configurable distance metric
- RUNG_combined: Cosine + percentile gamma (best stacking)

---

## Example: Full 4-Model Comparison (Recommended)

```bash
# Quick test (small budgets, single dataset)
python run_all.py --datasets cora \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined \
  --budgets 0.1 0.2 0.4

# Full comparison (all budgets, both datasets) — ~30 min on GPU
python run_all.py --datasets cora citeseer \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined

# With custom hyperparameters
python run_all.py --datasets cora citeseer \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined \
  --percentile_q 0.80 \
  --max_epoch 400 \
  --budgets 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
```

---

## Troubleshooting

### "attack.py: unknown arguments"
- Older version of `attack.py` doesn't support `--budgets`
- Solution: Update to latest version or pass single budget only

### "train_test_combined.py: command not found"
- Ensure `train_test_combined.py` exists and is executable
- Check path: `/home/ttt/Documents/RUNG_ML_Project/train_test_combined.py`

### Model training fails
- Check `log/<dataset>/clean/<model>.log` for errors
- Ensure GPU memory is sufficient for all 4 models
- Try `--skip_clean` if logs already exist, then `--skip_attack` to run only attacks

### "No logs generated"
- Check that `log/` directory exists and is writable
- Verify dataset files exist in `data/`
- Look for error messages in console output

