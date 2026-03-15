# Adversarial Training - Troubleshooting & FAQ

## Frequently Asked Questions

### Q: How do I get started?
**A:** Start with the quick demo:
```bash
cd /home/ttt/Documents/RUNG_ML_Project
python run_adversarial_quick_demo.py
```
This tests both models in ~1-2 minutes. Then check results:
```bash
cat exp/results/adversarial_demo/logs/results.json
```

### Q: What's the difference between the quick demo and full suite?
**A:**

| Feature | Quick Demo | Full Suite |
|---------|-----------|-----------|
| Time | 1-2 min | 10-30 min |
| Configs | 2 | 7 |
| File I/O | Minimal | Detailed logging |
| Best for | Verification | Ablation studies |

**Choose:**
- Quick demo: "Does everything work?"
- Full suite: "What are the best hyperparameters?"

### Q: How is adversarial training different from regular training?
**A:** Three main differences:

| Aspect | Normal | Adversarial |
|--------|--------|-------------|
| Loss | Only clean: L_clean | Mixed: 0.7×L_clean + 0.3×L_adv |
| Data | Fixed graphs | Perturbed graphs (every N epochs) |
| Robustness | Good on clean | Good on attacked graphs |

**Tradeoff:** Slower training but robust to adversarial attacks.

### Q: What is `alpha` and how should I set it?
**A:** Alpha controls the clean vs adversarial balance:

| Alpha | Training | Test Clean | Test Robust | Speed |
|-------|----------|-----------|------------|-------|
| 0.5 | Hard (more adv) | Lower | Higher | Slower |
| 0.7 | Balanced | Medium | Medium | Medium |
| 0.9 | Easy (more clean) | Higher | Lower | Faster |

**Recommendation:**
- Start with **0.7** (default, balanced)
- If test accuracy too low: increase to **0.8-0.9**
- If need more robustness: decrease to **0.5-0.6**

### Q: Why does training with adversarial examples take longer?
**A:** Because we do extra work per epoch:

```
Normal training per epoch:
  1. Forward pass (clean): model(A, X)
  2. Compute loss
  3. Backward + update

Adversarial training per epoch:
  1. Every 5 epochs: Generate attack A_adv = pgd_attack(model, A, X, y)  [Extra!]
  2. Forward pass (clean): model(A, X)
  3. Forward pass (adversarial): model(A_adv, X)                          [Extra!]
  4. Compute mixed loss (both models)
  5. Backward + update

So ~2x forward passes + periodically expensive attack generation.
```

**Speedup options:**
- Increase `--attack_freq` (5 → 10, regenerate every 10 epochs)
- Increase `--adv_alpha` (0.7 → 0.9, emphasize clean training)
- Decrease `--train_pgd_steps` (20 → 10, weaker attacks)

### Q: Can I use adversarial training with different datasets?
**A:** Yes! The system works with any dataset supported by RUNG:

```bash
# Cora (default)
python clean.py --model RUNG_percentile_adv --data cora --max_epoch 300

# Citeseer
python clean.py --model RUNG_percentile_adv --data citeseer --max_epoch 300

# Heterophilic (squirrel, chameleon)
python clean.py --model RUNG_percentile_adv --data squirrel --max_epoch 300
python clean.py --model RUNG_percentric_adv --data chameleon --max_epoch 300
```

### Q: How do I compare adversarial vs non-adversarial?
**A:** Run side-by-side:

```bash
# Non-adversarial baseline
python clean.py --model RUNG_percentile_gamma --data cora --max_epoch 300

# Adversarial training
python clean.py --model RUNG_percentile_adv --data cora --max_epoch 300

# Compare logs
ls -lh log/cora/
```

Compare test accuracy in generated tables.

### Q: What does "curriculum" mean in this context?
**A:** Curriculum = gradually increase attack strength during training:

```
Epoch   0-50      50-100    100-200    200+
Budget  5%        10%       20%        40%
        ▁▁▁▁▁     ░░░░░░░░░░ ▓▓▓▓▓▓▓▓▓▓▓▓▓ ██████████████████
```

**Why?** Model learns on easy attacks first, then hard. Improves convergence.

**Customize:** `--curriculum_budgets "5,15,30" --curriculum_epochs "40,80"`

### Q: What if training gets NaN losses?
**A:** Likely issues:

| Problem | Symptom | Fix |
|---------|---------|-----|
| Learning rate too high | Loss explodes | Decrease `--lr` (5e-2 → 1e-3) |
| Alpha out of range | Invalid loss weighting | Use `--adv_alpha 0.7` (0.1-1.0) |
| Gradient instability | NaN in gradients | Reduce `--train_pgd_steps` (20 → 5) |
| Model bug | Silent NaN propagation | Check model forward pass |

**Debug:**
```bash
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 5 --lr 1e-3 --seed 42 2>&1 | head -50
```

### Q: Can I use multiple GPUs?
**A:** Current implementation uses single GPU/CPU (DataParallel not yet integrated).

**To use GPU:**
```bash
# Should auto-detect
python clean.py --model RUNG_percentile_adv --data cora --max_epoch 300

# Or verify GPU usage
python -c "import torch; print(torch.cuda.is_available())"
```

**If CPU only:** Training will be slower but works fine for testing.

---

## Troubleshooting Guide

### Issue: Script Hangs / Gets Stuck

**Symptom:** Script starts but no output after 5+ minutes

**Cause:** 
- Generating initial attacks (expensive first time)
- Waiting for GPU memory
- Data loading

**Solution:**
```bash
# Kill it
Ctrl+C

# Run with smaller config first
python run_adversarial_quick_demo.py

# If that works, issue is memory/computation, use:
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 5 --train_pgd_steps 5 --attack_freq 10
```

### Issue: Out of Memory Error

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Cause:** GPU memory exhausted during training

**Solution (Priority Order):**
```bash
# 1. Reduce max_epoch
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 50 --lr 0.05

# 2. Reduce PGD steps
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 100 --train_pgd_steps 5

# 3. Increase attack frequency (fewer regenerations)
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 100 --attack_freq 20

# 4. Enable gradient accumulation (if supported)
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 100 --batch_size 32

# 5. Use CPU (much slower but works)
# Set CUDA_VISIBLE_DEVICES=""
CUDA_VISIBLE_DEVICES="" python clean.py --model RUNG_percentile_adv --data cora --max_epoch 50
```

### Issue: Very Slow Training

**Symptom:** One epoch takes > 2 minutes on Cora

**Cause:**
- High `train_pgd_steps` (expensive attacks)
- Low `attack_freq` (regenerating too often)
- Low `adv_alpha` (more adversarial processing)
- CPU training instead of GPU

**Solution:**
```bash
# Check GPU usage
nvidia-smi

# If GPU not used:
# - Reinstall PyTorch with CUDA
# - Or explicitly try: CUDA_VISIBLE_DEVICES=0 python clean.py ...

# Speed up training:
python clean.py --model RUNG_percentile_adv --data cora \
    --attack_freq 10 \
    --train_pgd_steps 10 \
    --adv_alpha 0.8
```

### Issue: Test Results Show Lower Accuracy Than Baseline

**Symptom:** `RUNG_percentile_adv` test accuracy < `RUNG_percentile_gamma`

**Cause:** Normal tradeoff - adversarial training sacrifices clean accuracy for robustness

**Is It a Problem?**
- **1-5% drop**: Normal for alpha=0.7
- **>10% drop**: Possible issues
  - Alpha too low (use 0.8-0.9)
  - Not enough epochs (increase max_epoch)
  - Learning rate too high

**Fix:**
```bash
# Try adjusting alpha
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 300 --adv_alpha 0.85

# Or increase training time
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 500 --adv_alpha 0.7
```

### Issue: Can't Import Modules

**Symptom:**
```
ModuleNotFoundError: No module named 'train_eval_data'
```

**Cause:** Running from wrong directory

**Solution:**
```bash
# Make sure you're in project root
cd /home/ttt/Documents/RUNG_ML_Project

# Then run
python clean.py --model RUNG_percentile_adv --data cora --max_epoch 10
```

### Issue: pgd_attack Not Found

**Symptom:**
```
ImportError: cannot import name 'pgd_attack' from 'experiments.run_ablation'
```

**Cause:** 
- run_ablation.py not in PYTHONPATH
- pgd_attack function doesn't exist

**Solution:**
```bash
# Check if file exists
ls -la experiments/run_ablation.py

# Check if function is defined
grep "def pgd_attack" experiments/run_ablation.py

# If not found, need to debug experiments/run_ablation.py
```

### Issue: Results Directory Not Created

**Symptom:**
```
No output in exp/results/adversarial_training/
```

**Cause:**
- Permissions issue
- Script failed before logging

**Solution:**
```bash
# Create directory manually
mkdir -p exp/results/adversarial_training/logs

# Try running again with verbose output
python run_adversarial_test_suite.py 2>&1 | tee /tmp/test_log.txt

# Check log
cat /tmp/test_log.txt | head -50
```

### Issue: Curriculum Budgets Parse Error

**Symptom:**
```
Error parsing curriculum_budgets
```

**Cause:** Wrong format for arguments

**Solution:**
```bash
# WRONG - these will fail
python clean.py --model RUNG_percentile_adv --curriculum_budgets 5 10 20 40
python clean.py --model RUNG_percentile_adv --curriculum_budgets [5,10,20,40]

# RIGHT - use comma-separated string
python clean.py --model RUNG_percentile_adv --curriculum_budgets "5,10,20,40"
```

---

## Performance Tuning

### Slow Training? Use This Checklist

```bash
# 1. Check if GPU is available
✓ python -c "import torch; print(torch.cuda.is_available())"

# 2. Use less aggressive attacks
✓ python clean.py --model RUNG_percentile_adv --data cora \
    --train_pgd_steps 5 --attack_freq 10

# 3. Use higher alpha (safer = faster)
✓ python clean.py --model RUNG_percentile_adv --data cora \
    --adv_alpha 0.9

# 4. Reduce training time
✓ python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 50

# 5. Verify GPU is actually being used
✓ nvidia-smi (watch for GPU-Util %)
```

### Better Accuracy? Try This Tuning

```bash
# 1. More training
python clean.py --model RUNG_percentile_adv --data cora --max_epoch 500

# 2. Reduce adversarial weight (easier optimization)
python clean.py --model RUNG_percentile_adv --data cora \
    --adv_alpha 0.85 --max_epoch 300

# 3. Stronger attacks
python clean.py --model RUNG_percentile_adv --data cora \
    --train_pgd_steps 50 --max_epoch 300

# 4. Finer curriculum
python clean.py --model RUNG_percentile_adv --data cora \
    --curriculum_budgets "3,7,15,30" --curriculum_epochs "40,80,160"
```

### More Robust To Attacks? Try This

```bash
# 1. Lower alpha (emphasize adversarial)
python clean.py --model RUNG_percentile_adv --data cora \
    --adv_alpha 0.5 --max_epoch 300

# 2. Stronger training attacks
python clean.py --model RUNG_percentile_adv --data cora \
    --train_pgd_steps 50 --max_epoch 300

# 3. More frequent attack regeneration
python clean.py --model RUNG_percentile_adv --data cora \
    --attack_freq 2 --max_epoch 300

# 4. Larger budget curves
python clean.py --model RUNG_percentile_adv --data cora \
    --curriculum_budgets "10,25,50,80" --curriculum_epochs "50,100,200"
```

---

## Debug Mode

### Enable Verbose Logging

```bash
# See detailed training info
python -u clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 5 --seed 42 2>&1 | tee /tmp/debug.log

# Then inspect
cat /tmp/debug.log | grep -i "loss\|accuracy\|epoch"
```

### Check Individual Components

```bash
# 1. Test data loading
python -c "
from train_eval_data import get_dataset
A, X, y, train_idx, val_idx, test_idx = get_dataset('cora')
print(f'A shape: {A.shape}, X shape: {X.shape}, y shape: {y.shape}')
print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
"

# 2. Test model creation
python -c "
from exp.config import get_model
model = get_model('RUNG_percentile_adv', 1433, 7)
print(model)
"

# 3. Test attack function
python -c "
from experiments.run_ablation import pgd_attack
print(f'pgd_attack signature: {pgd_attack.__doc__}')
"

# 4. Test curriculum
python -c "
from train_eval_data.adversarial_trainer import CurriculumSchedule
schedule = CurriculumSchedule()
for epoch in [0, 25, 50, 100, 200, 300]:
    print(f'Epoch {epoch}: phase={schedule.get_phase(epoch)}, budget={schedule.get_budget(epoch)}%')
"
```

### Profile Training Speed

```bash
# Time a single epoch
time python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 1 --train_pgd_steps 20

# Time without attacks
time python clean.py --model RUNG_percentile_gamma --data cora \
    --max_epoch 1

# Compare for speedup potential
```

---

## Common Configuration Recipes

### Recipe 1: Quick Testing
```bash
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 5 \
    --train_pgd_steps 5 \
    --attack_freq 10 \
    --adv_alpha 0.7
```
**Time:** ~10 seconds

### Recipe 2: Balanced Robustness
```bash
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 300 \
    --train_pgd_steps 20 \
    --attack_freq 5 \
    --adv_alpha 0.7
```
**Time:** ~30 minutes

### Recipe 3: Maximum Robustness
```bash
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 500 \
    --train_pgd_steps 50 \
    --attack_freq 2 \
    --adv_alpha 0.5 \
    --curriculum_budgets "10,25,50,80"
```
**Time:** ~60+ minutes

### Recipe 4: Fast Training (Accuracy Focus)
```bash
python clean.py --model RUNG_percentile_adv --data cora \
    --max_epoch 100 \
    --train_pgd_steps 5 \
    --attack_freq 20 \
    --adv_alpha 0.9
```
**Time:** ~5 minutes

---

## Getting Help

### Check Documentation
- **Quick reference:** [ADVERSARIAL_QUICK_REFERENCE.md](ADVERSARIAL_QUICK_REFERENCE.md)
- **Full guide:** [ADVERSARIAL_TEST_SUITE_README.md](ADVERSARIAL_TEST_SUITE_README.md)
- **Architecture:** [ADVERSARIAL_ARCHITECTURE.md](ADVERSARIAL_ARCHITECTURE.md)
- **Technical design:** [docs/changes/013_adversarial_training.md](docs/changes/013_adversarial_training.md)

### Run Diagnostic Script
```bash
python -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'GPU available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')

# Try imports
try:
    from train_eval_data import fit_percentile_adv
    from experiments.run_ablation import pgd_attack
    print('✓ All imports successful')
except Exception as e:
    print(f'✗ Import error: {e}')
"
```

---

**Last Updated:** March 2026  
**Version:** 1.0  
**Contributors:** Adversarial Training Implementation Team
