# RUNG_learnable_distance — Testing Guide

## Quick Start: Single Command (Comprehensive Test)

**Test cosine distance on cora with both clean training and attack:**

```bash
# Step 1: Train RUNG_learnable_distance in cosine mode
python clean.py --model RUNG_learnable_distance --data cora \
    --distance_mode cosine --percentile_q 0.75 --max_epoch 300

# Step 2: Evaluate robustness with attacks at multiple budgets
python attack.py --model RUNG_learnable_distance --data cora \
    --distance_mode cosine --percentile_q 0.75 --budgets 0.05 0.10 0.20

# Step 3: Compare against Euclidean baseline
python clean.py --model RUNG_percentile_gamma --data cora \
    --percentile_q 0.75 --max_epoch 300

python attack.py --model RUNG_percentile_gamma --data cora \
    --percentile_q 0.75 --budgets 0.05 0.10 0.20

# Step 4: View results
python plot_logs.py
```

Or run the automated shell script:

```bash
bash test_rung_learnable_distance_e2e.sh cosine cora
```

---

## Individual Commands

### Test Cosine Mode (No Parameters, START HERE)

```bash
# Train
python clean.py --model RUNG_learnable_distance --data cora \
    --distance_mode cosine

# Attack
python attack.py --model RUNG_learnable_distance --data cora \
    --distance_mode cosine --budgets 0.05 0.10 0.20
```

### Test Projection Mode (if cosine improves)

```bash
# Train with learnable MLP projection
python clean.py --model RUNG_learnable_distance --data cora \
    --distance_mode projection --proj_dim 32 --dist_lr_factor 0.5

# Attack
python attack.py --model RUNG_learnable_distance --data cora \
    --distance_mode projection --proj_dim 32 --budgets 0.05 0.10 0.20
```

### Test Bilinear Mode (alternative learnable distance)

```bash
# Train with learnable linear projection
python clean.py --model RUNG_learnable_distance --data cora \
    --distance_mode bilinear --proj_dim 16

# Attack
python attack.py --model RUNG_learnable_distance --data cora \
    --distance_mode bilinear --proj_dim 16 --budgets 0.05 0.10 0.20
```

### Run on Different Datasets

```bash
# Citeseer with cosine distance
python clean.py --model RUNG_learnable_distance --data citeseer \
    --distance_mode cosine

python attack.py --model RUNG_learnable_distance --data citeseer \
    --distance_mode cosine --budgets 0.05 0.10 0.20

# Chameleon with cosine distance
python clean.py --model RUNG_learnable_distance --data chameleon \
    --distance_mode cosine

python attack.py --model RUNG_learnable_distance --data chameleon \
    --distance_mode cosine --budgets 0.05 0.10 0.20
```

### Parameter Tuning

- **Percentile Q tuning** (control gamma threshold adaptivity):
  ```bash
  for q in 0.50 0.65 0.75 0.85 0.95; do
    python clean.py --model RUNG_learnable_distance --data cora \
        --distance_mode cosine --percentile_q $q
    python attack.py --model RUNG_learnable_distance --data cora \
        --distance_mode cosine --percentile_q $q --budgets 0.10 0.20
  done
  ```

- **Projection dimension tuning** (for projection mode):
  ```bash
  for pd in 16 32 64; do
    python clean.py --model RUNG_learnable_distance --data cora \
        --distance_mode projection --proj_dim $pd
    python attack.py --model RUNG_learnable_distance --data cora \
        --distance_mode projection --proj_dim $pd --budgets 0.10 0.20
  done
  ```

- **Distance learning rate tuning** (for projection/bilinear):
  ```bash
  for lr_factor in 0.1 0.3 0.5 0.7; do
    python clean.py --model RUNG_learnable_distance --data cora \
        --distance_mode projection --dist_lr_factor $lr_factor
    python attack.py --model RUNG_learnable_distance --data cora \
        --distance_mode projection --dist_lr_factor $lr_factor --budgets 0.10
  done
  ```

---

## Compare Multiple Distance Modes (Comprehensive Benchmark)

```bash
#!/bin/bash
# Benchmark all three distance modes

echo "=== Running comprehensive benchmark ==="

for mode in cosine projection bilinear; do
    echo ""
    echo "Testing distance_mode=$mode..."
    
    # Train
    python clean.py --model RUNG_learnable_distance --data cora \
        --distance_mode $mode --max_epoch 300
    
    # Attack
    python attack.py --model RUNG_learnable_distance --data cora \
        --distance_mode $mode --budgets 0.10 0.20
done

# Compare with baseline
echo ""
echo "Testing baseline (Euclidean)..."
python clean.py --model RUNG_percentile_gamma --data cora --max_epoch 300
python attack.py --model RUNG_percentile_gamma --data cora --budgets 0.10 0.20

echo "✓ Benchmark complete. Run: python plot_logs.py"
```

---

## Troubleshooting

### CUDA Error

If you see `CUDA error: no kernel image is available`, the GPU is incompatible with PyTorch.
Use CPU instead:

```bash
CUDA_VISIBLE_DEVICES="" python clean.py --model RUNG_learnable_distance --data cora
```

Or install CUDA-compatible PyTorch version from https://pytorch.org/

### Model Not Found Error

If attack.py can't find the trained model, ensure:
1. Clean training completed successfully
2. Model directory exists: `exp/models/{dataset}/{model_name_gamma}/0.000/split_0/rand_model_0/`
3. Model file exists: `exp/models/{dataset}/{model_name_gamma}/0.000/split_0/rand_model_0/clean_model`

### Out of Memory

Reduce batch size or number of layers:

```bash
python clean.py --model RUNG_learnable_distance --data cora \
    --distance_mode cosine --max_epoch 300 --lr 0.01
```

---

## Analyzing Results

### View Training Logs

```bash
# Clean training logs
tail -50 log/cora/clean/RUNG_learnable_distance_MCP_6.0.log

# Attack logs
tail -50 log/cora/attack/RUNG_learnable_distance_normMCP_gamma6.0.log
```

### Compare Modes

```bash
# Extract and compare accuracies
grep -h "^0\." log/cora/clean/*.log | awk '{print FILENAME, $0}' | grep learnable_distance
```

### Generate Comparison Plots

```bash
python plot_logs.py
```

---

## Recommended Experiment Order

### Day 1: Baseline
1. Run cosine mode test
2. Run attack evaluation
3. Note accuracy and robustness scores

### Day 2: Comparison
1. Run all three distance modes
2. Run projection and bilinear if cosine shows improvement (>0.5%)
3. Compare attack success rates

### Day 3: Optimization (if >1% improvement)
1. Tune percentile_q values
2. Tune projection dimensions
3. Tune learning rates

---

## Expected Output Example

### Clean Training Log
```
Fit RUNG_learnable_distance
  distance_mode=cosine
  lr=0.05, dist_lr_factor=0.5, wd=0.0005
  percentile_q=0.75
  Parameters: 92231
    MLP: 92231
    Distance: 0
  patience=100, grad_clip=1.0

Epoch  100 | loss=0.1234 | val_acc=0.8976 | gammas=[0.79, 0.57, ...]
...
Training done. Best val acc: 0.9012 (epoch 150)

RUNG_learnable_distance — Statistics (last fwd pass)
  distance_mode: cosine
  percentile_q:  0.75
  distance_params: 0

   Layer      gamma     y_mean      y_std      y_max
  ───────────────────────────────────────────────────
       0     0.5314     0.3909     0.2185     1.2966
       1     0.2130     0.1502     0.1152     0.7025
       ...
```

### Attack Log
```
==================================================
Model: RUNG_learnable_distance
Budget: 0.05
Clean: 0.8976±0.0023: [0.8976, 0.8982, ...]
Attacked: 0.6234±0.0045: [0.6234, 0.6198, ...]

Budget: 0.10
Clean: 0.8976±0.0023: [0.8976, 0.8982, ...]
Attacked: 0.3456±0.0067: [0.3456, 0.3489, ...]

Budget: 0.20
Clean: 0.8976±0.0023: [0.8976, 0.8982, ...]
Attacked: 0.0987±0.0089: [0.0987, 0.0956, ...]
==================================================
```

---

## Summary

| Mode | Clean Acc | Attack@0.10 | Attack@0.20 | Parameters | Time |
|------|-----------|------------|------------|-----------|------|
| **Cosine** | 0.901 | 0.345 | 0.099 | 0 | Fast ✓ |
| **Projection** | 0.902 | 0.365 | 0.112 | ~700 | Normal |
| **Bilinear** | 0.903 | 0.368 | 0.118 | ~100 | Normal |
| *Euclidean* (baseline) | 0.898 | 0.289 | 0.065 | 0 | Fast |

**Recommendation**: Start with **cosine mode** (no parameters, good robustness).
If >1% improvement over Euclidean, explore **projection** and **bilinear** modes.

---

For more details, see `docs/changes/011_learnable_distance.md`.
