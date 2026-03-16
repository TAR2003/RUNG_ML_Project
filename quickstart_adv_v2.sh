#!/bin/bash
# quickstart_adv_v2.sh
# Quick start guide for RUNG_percentile_adv_v2

set -e

echo "=============================================================================="
echo "RUNG_percentile_adv_v2 — Quick Start Guide"
echo "=============================================================================="
echo ""
echo "This script helps you get started with the fixed adversarial training."
echo ""
echo "Before running this, read: docs/changes/014_adversarial_v2.md"
echo ""

# Check if running from the right directory
if [ ! -f "clean.py" ]; then
    echo "ERROR: This script must be run from project root"
    echo "Usage: cd /path/to/RUNG_ML_Project && bash quickstart_adv_v2.sh"
    exit 1
fi

echo "=============================================================================="
echo "STEP 1: Verify Attack is Adaptive"
echo "=============================================================================="
echo ""
echo "Running diagnostic to check if pgd_attack computes gradients correctly..."
echo "This is a PREREQUISITE for adversarial training to work."
echo ""

python diagnose_attack.py --dataset cora

echo ""
echo "If you saw 'DIAGNOSIS: ATTACK IS ADAPTIVE ✓' then proceed to Step 2."
echo "If you saw 'DIAGNOSIS: ATTACK IS NOT ADAPTIVE' then STOP."
echo "Fix the attack function first (check experiments/run_ablation.py::pgd_attack)"
echo ""
read -p "Press Enter to continue with Step 2..."

echo ""
echo "=============================================================================="
echo "STEP 2: Check Attack Strength"
echo "=============================================================================="
echo ""
echo "Verifying that training attack strength matches test attack strength..."
echo "This prevents the model from training against weak attacks and failing strong ones."
echo ""

python diagnose_attack_strength.py --model RUNG_percentile_gamma --dataset cora

echo ""
echo "Key metric: Gap (20-step vs 200-step)"
echo "  - Gap < 5%: GOOD — attacks well matched"
echo "  - Gap > 5%: BAD — consider increasing training steps"
echo ""
read -p "Press Enter to continue with Step 3..."

echo ""
echo "=============================================================================="
echo "STEP 3a: QUICK TEST (for development/debugging)"
echo "=============================================================================="
echo ""
echo "Training for 200 epochs with 50 steps (instead of 100)."
echo "This takes ~30-45 minutes on GPU for faster iteration."
echo "Results will be suboptimal but useful for debugging."
echo ""
echo "Command: python train_and_test_adv_v2.py --dataset cora --max_epoch 200 --train_pgd_steps 50 --warmup_epochs 50"
echo ""
read -p "Run quick test? (y/n) " -n 1 -r; echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train_and_test_adv_v2.py --dataset cora \
                                   --max_epoch 200 \
                                   --train_pgd_steps 50 \
                                   --warmup_epochs 50
fi

echo ""
echo "=============================================================================="
echo "STEP 3b: FULL TRAINING (recommended)"
echo "=============================================================================="
echo ""
echo "Full v2 training with recommended parameters:"
echo "  - 800 epochs"
echo "  - 100 PGD steps (strong)"
echo "  - alpha=0.85 (gentle adversarial)"
echo "  - 100 warmup epochs (stabilization)"
echo ""
echo "This will take 3-6 hours on GPU."
echo ""
echo "Command: python train_and_test_adv_v2.py --dataset cora"
echo ""
read -p "Run full training? (y/n) " -n 1 -r; echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train_and_test_adv_v2.py --dataset cora
fi

echo ""
echo "=============================================================================="
echo "STEP 4: Compare with V1"
echo "=============================================================================="
echo ""
echo "Optional: Train the original (unfixed) v1 for comparison."
echo "This will show you the improvement from the fixes."
echo ""
echo "Command: python clean.py --model RUNG_percentile_adv --data cora --max_epoch 800"
echo ""
read -p "Run v1 training? (y/n) " -n 1 -r; echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python clean.py --model RUNG_percentile_adv --data cora --max_epoch 800
fi

echo ""
echo "=============================================================================="
echo "Done!"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Check results in console output above"
echo "  2. If issues, refer to docs/changes/014_adversarial_v2.md#Common Issues"
echo "  3. For detailed info, read: docs/changes/014_adversarial_v2.md"
echo ""
