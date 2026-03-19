#!/bin/bash
# test_rung_learnable_distance_e2e.sh
#
# End-to-end test: train RUNG_learnable_distance and evaluate robustness
#
# Usage:
#   bash test_rung_learnable_distance_e2e.sh [cosine|projection|bilinear] [cora|citeseer]
#

set -e  # Exit on error

DISTANCE_MODE="${1:-cosine}"
DATASET="${2:-cora}"
MAX_EPOCHS=300
PATIENCE=100
BUDGETS="0.05 0.10 0.20"

echo "========================================================================"
echo "RUNG_learnable_distance End-to-End Test"
echo "========================================================================"
echo "Distance Mode: $DISTANCE_MODE"
echo "Dataset:      $DATASET"
echo "Max Epochs:   $MAX_EPOCHS"
echo "Budgets:      $BUDGETS"
echo ""

# ========================================================================
# PHASE 1: Clean Training
# ========================================================================
echo "[PHASE 1] Training RUNG_learnable_distance with distance_mode=$DISTANCE_MODE"
echo "------------------------------------------------------------------------"

python clean.py \
    --model RUNG_learnable_distance \
    --data "$DATASET" \
    --distance_mode "$DISTANCE_MODE" \
    --percentile_q 0.75 \
    --max_epoch "$MAX_EPOCHS" \
    --lr 0.05 \
    --weight_decay 5e-4

echo "✓ Clean training complete"
echo ""

# ========================================================================
# PHASE 2: Adversarial Attack Evaluation
# ========================================================================
echo "[PHASE 2] Evaluating robustness with PGD attacks"
echo "------------------------------------------------------------------------"

python attack.py \
    --model RUNG_learnable_distance \
    --data "$DATASET" \
    --distance_mode "$DISTANCE_MODE" \
    --percentile_q 0.75 \
    --budgets $BUDGETS

echo "✓ Attack evaluation complete"
echo ""

# ========================================================================
# PHASE 3: Comparison with Baseline
# ========================================================================
echo "[PHASE 3] Running baseline (RUNG_percentile_gamma with Euclidean distance)"
echo "------------------------------------------------------------------------"

echo "Training baseline..."
python clean.py \
    --model RUNG_percentile_gamma \
    --data "$DATASET" \
    --percentile_q 0.75 \
    --max_epoch "$MAX_EPOCHS" \
    --lr 0.05 \
    --weight_decay 5e-4

echo "Attacking baseline..."
python attack.py \
    --model RUNG_percentile_gamma \
    --data "$DATASET" \
    --percentile_q 0.75 \
    --budgets $BUDGETS

echo "✓ Baseline evaluation complete"
echo ""

# ========================================================================
# Summary
# ========================================================================
echo "========================================================================"
echo "✓ ALL TESTS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - Logs: log/$DATASET/clean/"
echo "  - Logs: log/$DATASET/attack/"
echo ""
echo "To visualize results, run:"
echo "  python plot_logs.py"
echo ""
