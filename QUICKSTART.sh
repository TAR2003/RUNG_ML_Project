#!/bin/bash
# QUICK START GUIDE FOR RUNG_COMBINED

# ============================================================================
# RUNG_COMBINED - Quick Start Commands
# ============================================================================

# ============================================================================
# STEP 1: VERIFY EVERYTHING WORKS (2-3 minutes)
# ============================================================================
echo "=== STEP 1: Verification ==="
python verify_rung_combined.py
# Expected output: "ALL VERIFICATION TESTS PASSED ✓"


# ============================================================================
# STEP 2: QUICK TRAINING TEST (5-10 minutes, learning check)
# ============================================================================
echo "=== STEP 2: Quick Training Test ==="
python train_test_combined.py --dataset cora --max_epoch 50 --skip_attack
# Tests: model trains, no crashes
# Output: Clean test accuracy on Cora


# ============================================================================
# STEP 3: FULL TRAINING + ATTACK ON CORA (30-45 minutes)
# ============================================================================
echo "=== STEP 3: Full Cora Experiment ==="
python train_test_combined.py --dataset cora --max_epoch 300 --lr 0.05
# Trains model completely + tests with all budgets [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
# Output: Complete results table with clean + attacked accuracies


# ============================================================================
# STEP 4: TUNE PERCENTILE_Q (Optional, 2-3 hours for full search)
# ============================================================================
echo "=== STEP 4: Percentile Q Tuning ==="
for q in 0.60 0.65 0.70 0.75 0.80 0.85; do
    echo "Testing q=$q..."
    python train_test_combined.py --dataset cora --percentile_q $q \
        --max_epoch 200 --skip_attack
    # Outputs test accuracy for this q value
done
# Pick q with best results


# ============================================================================
# STEP 5: MULTI-DATASET EXPERIMENT (1-2 hours)
# ============================================================================
echo "=== STEP 5: All Datasets ==="
python train_test_combined.py --datasets cora citeseer
# Trains on both datasets sequentially


# ============================================================================
# STEP 6: COMPARE WITH PARENT MODELS (3-4 hours)
# ============================================================================
echo "=== STEP 6: Comparison ==="
echo "RUNG_percentile_gamma:"
python train_test_combined.py --dataset cora --max_epoch 300 --skip_attack
# Switch model in script if needed

echo "RUNG_learnable_distance:"
python attack.py --model RUNG_learnable_distance --data cora \
    --distance_mode cosine --budgets 0.05 0.10 0.20 0.30 0.40 0.60

echo "RUNG_combined:"
python train_test_combined.py --dataset cora --max_epoch 300
# Compare results side-by-side


# ============================================================================
# ADVANCED USAGE
# ============================================================================

# Train with custom hyperparameters
python train_test_combined.py --dataset cora \
    --max_epoch 500 \
    --lr 0.01 \
    --percentile_q 0.70 \
    --weight_decay 1e-4

# Skip attack for faster training (e.g., for hyperparameter search)
python train_test_combined.py --dataset cora --max_epoch 100 --skip_attack

# Use with existing pipeline
python clean.py --model RUNG_combined --data cora
python attack.py --model RUNG_combined --data cora --budgets 0.05 0.10 0.20 0.30 0.40 0.60

# Verify model works with run_all.py
python run_all.py --datasets cora --models RUNG_combined --max_epoch 300


# ============================================================================
# EXPECTED TIMING
# ============================================================================
# Verification:          2-3 min
# Quick training (50ep): 5-10 min
# Full training (300ep): 30-45 min
# Per-budget attack (6): ~5 min each = 30 min total
# Full experiment:       60-90 min per dataset


# ============================================================================
# EXPECTED RESULTS (Cora dataset)
# ============================================================================
# Clean test accuracy:   ~82% (between parents: 81% percentile, 77% distance)
# Accuracy @ budget 0.40: ~76% (between parents: 75% percentile, 71% distance)
# Variance:              Low (stable across seeds)


# ============================================================================
# DEBUGGING
# ============================================================================

# If model doesn't run:
python verify_rung_combined.py  # Check what fails

# If training is slow:
python train_test_combined.py --dataset cora --max_epoch 10 --skip_attack

# If attacks fail:
python attack.py --model RUNG_combined --data cora --budgets 0.05

# Check GPU availability:
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Use CPU explicitly (no GPU compatibility issues):
python train_test_combined.py --dataset cora --device cpu


# ============================================================================
# SAVING RESULTS
# ============================================================================

# Results are printed to stdout and can be saved:
python train_test_combined.py --dataset cora > results_cora.txt 2>&1

# Parse results from log file:
grep -E "accuracy|epoch|budget" results_cora.txt


# ============================================================================
# ANALYSIS
# ============================================================================

# Create summary table from multiple runs:
for dataset in cora citeseer; do
    echo "=== $dataset ==="
    python train_test_combined.py --datasets $dataset --max_epoch 300 | grep -E "accuracy|Clean"
done

# Compare parameter counts:
python -c "
from exp.config.get_model import get_model_default
for model_name in ['RUNG', 'RUNG_percentile_gamma', 'RUNG_learnable_distance', 'RUNG_combined']:
    m, _ = get_model_default('cora', model_name, device='cpu')
    print(f'{model_name:<30} {m.count_parameters():>8} params')
"
