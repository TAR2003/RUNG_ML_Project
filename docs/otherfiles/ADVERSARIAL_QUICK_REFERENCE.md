# Adversarial Training - Quick Reference Card

## One-Command Execution

```bash
# Quick test (1-2 minutes)
python run_adversarial_quick_demo.py

# Full ablation suite (10-30 minutes)
python run_adversarial_test_suite.py
```

## Models Available

```bash
# Percentile-based RUNG with adversarial training
python clean.py --model RUNG_percentile_adv --data cora --max_epoch 300

# Parametric-based RUNG with adversarial training
python clean.py --model RUNG_parametric_adv --data citeseer --max_epoch 300
```

## Key Hyperparameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--adv_alpha` | 0.7 | Clean loss weight: higher = safer/faster, lower = stronger robustness |
| `--attack_freq` | 5 | Regenerate adversarial examples every N epochs |
| `--train_pgd_steps` | 20 | PGD attack iterations during training |
| `--curriculum_budgets` | "5,10,20,40" | Edge budget % per phase |
| `--curriculum_epochs` | "50,100,200" | Epoch boundaries between phases |

## Full Training Examples

### Baseline Adversarial Training (Cora)
```bash
python clean.py \
    --model RUNG_percentile_adv \
    --data cora \
    --max_epoch 300 \
    --adv_alpha 0.7 \
    --attack_freq 5
```

### More Robust (Lower Alpha)
```bash
python clean.py \
    --model RUNG_percentile_adv \
    --data cora \
    --max_epoch 300 \
    --adv_alpha 0.5  # More adversarial training
```

### Faster Training (Higher Attack Freq)
```bash
python clean.py \
    --model RUNG_percentile_adv \
    --data cora \
    --max_epoch 300 \
    --attack_freq 10  # Fewer attack regenerations
```

### Stronger Attacks (Higher PGD Steps)
```bash
python clean.py \
    --model RUNG_parametric_adv \
    --data citeseer \
    --max_epoch 300 \
    --train_pgd_steps 50  # More expensive attacks
```

## Output Locations

| Script | Results Directory |
|--------|-------------------|
| `run_adversarial_quick_demo.py` | `exp/results/adversarial_demo/logs/` |
| `run_adversarial_test_suite.py` | `exp/results/adversarial_training/logs/` |
| `clean.py` training | `log/{dataset}/{model}/` |

## View Results

```bash
# Quick demo results
cat exp/results/adversarial_demo/logs/results.json

# Full suite summary
cat exp/results/adversarial_training/report.txt

# View JSON summary (machine-readable)
cat exp/results/adversarial_training/report.json

# Last 30 lines of latest training log
tail -30 exp/results/adversarial_training/logs/*.log | tail -30

# Count successful runs
grep "✓" exp/results/adversarial_training/report.txt | wc -l
```

## Understanding Alpha Parameter

```
Alpha = 0.5  →  More Adversarial (50% clean, 50% adversarial)
               Slower, stronger robustness, harder to optimize

Alpha = 0.7  →  Balanced (70% clean, 30% adversarial) [DEFAULT]
               Good tradeoff between clean & robust performance

Alpha = 0.9  →  More Clean (90% clean, 10% adversarial)
               Faster, clean-focused, less robust
```

## Loss During Training

$$L_{total} = \alpha \cdot L_{clean} + (1-\alpha) \cdot L_{adv}$$

- $L_{clean}$: Standard training loss on clean data
- $L_{adv}$: Loss on adversarially perturbed graphs
- $\alpha$: Balance weight (default 0.7)

## Curriculum Budget Schedule

Default: Budget increases over training phases
- **Phase 1** (0-50 epochs): 5% edges can be attacked
- **Phase 2** (50-100 epochs): 10% edges can be attacked
- **Phase 3** (100-200 epochs): 20% edges can be attacked
- **Phase 4** (200+ epochs): 40% edges can be attacked

Customize with `--curriculum_budgets "5,10,15,25"` and `--curriculum_epochs "40,80,160"`

## Performance Tips

| Goal | Action |
|------|--------|
| Faster training | Increase `--attack_freq` (5 → 10) |
| Stronger robustness | Decrease `--adv_alpha` (0.7 → 0.5) |
| Better clean performance | Increase `--adv_alpha` (0.7 → 0.9) |
| Faster convergence | Increase `--lr` (5e-2 → 1e-1) |
| More stable training | Increase `--attack_freq` + decrease `--train_pgd_steps` |

## Debug Options

```bash
# Run with verbose logging
python clean.py --model RUNG_percentile_adv --data cora --max_epoch 10 --seed 42

# Test if everything loads
python -c "
from train_eval_data import fit_percentile_adv, fit_parametric_adv
from experiments.run_ablation import pgd_attack
print('✓ All modules imported successfully')
"

# Verify device setup
python -c "
import torch
print(f'GPU available: {torch.cuda.is_available()}')
print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')
"
```

## Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| Out of memory | Lower `max_epoch`, increase `--attack_freq` |
| Training too slow | Lower `--train_pgd_steps`, increase `--attack_freq` |
| High variance in loss | Reduce `--lr`, or ensure `--weight_decay` is set |
| NaN losses | Check that `--adv_alpha` is in [0.1, 1.0] |
| Script hangs | Press Ctrl+C, check logs for errors |

## Data Formats

All models expect:
- **A** (ndarray/tensor): Adjacency matrix [N×N]
- **X** (ndarray/tensor): Node features [N×D]
- **y** (ndarray/tensor): Node labels [N]
- **train_idx, val_idx, test_idx** (arrays): Node indices

Where N = # nodes, D = # features

## Citation

These adversarial training variants are built on RUNG models:
```
@inproceedings{rung,
  title={RUNG: Robustness via Uncertainty in Graph Neural Networks},
  ...
}
```

---

**Quick Links:**
- Full guide: [ADVERSARIAL_TEST_SUITE_README.md](ADVERSARIAL_TEST_SUITE_README.md)
- Implementation: [train_eval_data/adversarial_trainer.py](train_eval_data/adversarial_trainer.py)
- Testing: [run_adversarial_test_suite.py](run_adversarial_test_suite.py)
