# Adversarial Training Test Suite - Quick Start Guide

## Overview

Three scripts to test RUNG adversarial training models with automatic result saving:

| Script | Purpose | Time | Tests |
|--------|---------|------|-------|
| **`run_adversarial_quick_demo.py`** | Quick validation | ~1-2 min | 2 models, baseline config |
| **`run_adversarial_test_suite.py`** | Full ablation study | ~10-30 min | 7 configurations, hyperparameter sweeps |
| **`test_integration_final.py`** | Component verification | ~1-2 min | Technical checks only |

## Quick Start

### Run Quick Demo (Fastest - ~1-2 mins)
```bash
cd /home/ttt/Documents/RUNG_ML_Project

# Run 2 quick baseline tests
python run_adversarial_quick_demo.py

# Results saved to: exp/results/adversarial_demo/
```

**Output:**
```
[1/2] Demo 1: percentile_adv
  ✓ SUCCESS (33.2s)

[2/2] Demo 2: parametric_adv
  ✓ SUCCESS (34.1s)

Results saved to: exp/results/adversarial_demo/logs/
```

### Run Full Test Suite (Comprehensive - ~10-30 mins)
```bash
# Full ablation study: alpha sweep, frequency sweep, both models
python run_adversarial_test_suite.py

# Results saved to: exp/results/adversarial_training/
```

**Includes:**
- ✓ 2 baseline tests (percentile + parametric)
- ✓ 2 alpha sensitivity tests (alpha=0.5, 0.9)
- ✓ 2 attack frequency tests (freq=1, 10)
- ✓ 1 full parametric test

### View Individual Training Logs
```bash
# See all logs created
ls -lh exp/results/adversarial_training/logs/

# View specific test log (last 50 lines)
tail -50 exp/results/adversarial_training/logs/01_RUNG_percentile_adv_*.log
```

## Output Structure

### Demo Results
```
exp/results/adversarial_demo/
├── logs/
│   ├── 1_RUNG_percentile_adv_e2.log
│   └── 2_RUNG_parametric_adv_e2.log
└── results.json              # Summary in JSON
```

### Full Suite Results
```
exp/results/adversarial_training/
├── logs/
│   ├── 01_RUNG_percentile_adv_...log
│   ├── 02_RUNG_parametric_adv_...log
│   ├── 03_RUNG_percentile_adv_alpha0.5_...log
│   └── [... 7 total logs ...]
├── report.txt               # Human-readable summary
└── report.json              # Machine-readable summary
```

## Understanding Results

### Text Report Example (`report.txt`)
```
================================================================================
ADVERSARIAL TRAINING TEST SUITE - SUMMARY REPORT
================================================================================

Timestamp:      2026-03-16T12:34:56.789123
Total runs:     7
Successful:     7
Failed:         0
Total time:     245.3s

✓ [01] BASELINE: percentile_adv, quick (2 epochs)
       Time: 33.2s
       Alpha: 0.7, Freq: 5, Steps: 5, Epochs: 2

✓ [02] BASELINE: parametric_adv, quick (2 epochs)
       Time: 34.1s
       Alpha: 0.7, Freq: 5, Steps: 5, Epochs: 2

✓ [03] ALPHA: percentile_adv, alpha=0.5 (more adversarial)
       Time: 47.3s
       Alpha: 0.5, Freq: 5, Steps: 10, Epochs: 10

... [etc] ...
```

### JSON Report Example (`report.json`)
```json
{
  "timestamp": "2026-03-16T12:34:56.789123",
  "total_runs": 7,
  "successful_runs": 7,
  "failed_runs": 0,
  "total_time_seconds": 245.3,
  "results": [
    {
      "run": 1,
      "config": {
        "model": "RUNG_percentile_adv",
        "alpha": 0.7,
        "attack_freq": 5
      },
      "success": true,
      "elapsed_seconds": 33.2
    },
    ...
  ]
}
```

## Customizing Tests

### Modify `run_adversarial_test_suite.py`

To add or change test configurations, edit the `TEST_CONFIGS` list:

```python
TEST_CONFIGS = [
    {
        'model': 'RUNG_percentile_adv',      # Model name
        'dataset': 'cora',                    # Dataset
        'alpha': 0.7,                         # Clean loss weight (0.5-0.9)
        'attack_freq': 5,                     # Regenerate attack every N epochs
        'train_pgd_steps': 10,                # PGD steps during training
        'max_epoch': 100,                     # Training epochs
        'description': 'My custom config',    # Description for reporting
    },
    # Add more configurations...
]
```

### Example: Test Different Datasets
```python
TEST_CONFIGS = [
    {
        'model': 'RUNG_percentile_adv',
        'dataset': 'cora',
        'alpha': 0.7,
        'attack_freq': 5,
        'train_pgd_steps': 10,
        'max_epoch': 50,
        'description': 'Cora config',
    },
    {
        'model': 'RUNG_percentile_adv',
        'dataset': 'citeseer',  # Different dataset
        'alpha': 0.7,
        'attack_freq': 5,
        'train_pgd_steps': 10,
        'max_epoch': 50,
        'description': 'Citeseer config',
    },
]
```

### Example: Extended Alpha Sweep
```python
# Test alpha in {0.5, 0.6, 0.7, 0.8, 0.9}
TEST_CONFIGS = [
    {
        'model': 'RUNG_percentile_adv',
        'dataset': 'cora',
        'alpha': alpha,
        'attack_freq': 5,
        'train_pgd_steps': 10,
        'max_epoch': 50,
        'description': f'Alpha sensitivity: alpha={alpha}',
    }
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]
]
```

## Training Hyperparameter Reference

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--adv_alpha` | 0.7 | [0.5, 0.9] | Clean loss weight (higher = safer, slower robust gain) |
| `--attack_freq` | 5 | [1, 10] | Regenerate attack every N epochs (lower = slower, stronger) |
| `--train_pgd_steps` | 20 | [5, 50] | PGD iterations during training (higher = stronger, slower) |
| `--max_epoch` | 300 | [10, 1000] | Training epochs |
| `--lr` | 5e-2 | [1e-3, 1e-1] | Learning rate |
| `--weight_decay` | 5e-4 | [0, 1e-3] | L2 regularization |

## Advanced Usage

### Run Tests in Background
```bash
# Run in background, redirect output
nohup python run_adversarial_test_suite.py > /tmp/test_suite.log 2>&1 &

# Check progress
tail -f /tmp/test_suite.log

# Check PID
jobs
```

### Analyze Results
```bash
# View JSON results programmatically
python -c "
import json
with open('exp/results/adversarial_training/report.json') as f:
    data = json.load(f)
    print(f\"Total runs: {data['total_runs']}\")
    print(f\"Successful: {data['successful_runs']}\")
    print(f\"Time: {data['total_time_seconds']:.1f}s\")
"
```

### Compare Alpha Sensitivity
```bash
# View training time per alpha level
grep "alpha" exp/results/adversarial_training/report.txt | grep "Time:"
```

## What Each Test Does

### 1. Baseline Tests
- **Purpose**: Verify both models work end-to-end
- **Duration**: ~30-35s each
- **Epochs**: 2 (very short)
- **Config**: Standard (alpha=0.7, freq=5, steps=5)

### 2. Alpha Sensitivity Tests
- **Purpose**: Find optimal clean/robust tradeoff
- **Duration**: ~40-50s each
- **Epochs**: 10
- **Configs**: alpha=0.5 (more adversarial), alpha=0.9 (more clean)

### 3. Attack Frequency Tests
- **Purpose**: Runtime vs strength tradeoff
- **Duration**: ~20-40s each
- **Epochs**: 5
- **Configs**: freq=1 (every epoch), freq=10 (every 10 epochs)

### 4. Full Parametric Test
- **Purpose**: Verify parametric model with realistic config
- **Duration**: ~45s
- **Epochs**: 10
- **Config**: Full hyperparameter set

## Troubleshooting

### Script Hangs
```bash
# Kill it
Ctrl+C

# Or from another terminal
pkill -f run_adversarial
```

### Out of Memory
```bash
# Reduce max_epoch in TEST_CONFIGS
'max_epoch': 5,  # Instead of 50

# Or reduce training PGD steps
'train_pgd_steps': 5,  # Instead of 20
```

### Slow Training
- Increase `attack_freq` (regenerate less often)
- Reduce `train_pgd_steps` (fewer PGD iterations)
- Use smaller `max_epoch`

### Check Logs for Errors
```bash
# View log of failed test
cat exp/results/adversarial_training/logs/*FAILED*.log

# Search for errors
grep -i "error\|exception\|failed" exp/results/adversarial_training/logs/*.log
```

## Integration with run_all.py

To use adversarial models with `run_all.py`:

```bash
python run_all.py \
    --datasets cora citeseer \
    --models RUNG_percentile_adv RUNG_parametric_adv \
    --max_epoch 300 \
    --adv_alpha 0.7 \
    --attack_freq 5 \
    --skip_attack
```

## Next Steps

1. **Run quick demo** to verify everything works
2. **Check output** in `exp/results/adversarial_demo/logs/`
3. **Run full suite** for comprehensive testing
4. **Analyze results** in `exp/results/adversarial_training/report.txt`
5. **Train full models** for final evaluation with `clean.py` or `run_all.py`

---

**Created**: March 2026
**Script versions**: 
- `run_adversarial_quick_demo.py` - v1.0
- `run_adversarial_test_suite.py` - v1.0
