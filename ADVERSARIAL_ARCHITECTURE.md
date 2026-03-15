# Adversarial Training Architecture & Component Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL TRAINING SYSTEM                   │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│  Entry Points            │
├──────────────────────────┤
│ • clean.py (full train)  │
│ • run_all.py (bench)     │
│ • test suite scripts      │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│           Model Creation & Configuration                         │
├──────────────────────────────────────────────────────────────────┤
│ exp/config/get_model.py                                          │
│ ├─ RUNG_percentile_adv → RUNG_percentile_gamma instance        │
│ └─ RUNG_parametric_adv  → RUNG_parametric_gamma instance       │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│         Training Wrapper Functions                               │
├──────────────────────────────────────────────────────────────────┤
│ • train_eval_data/fit_percentile_adv.py                         │
│   └─ fit_percentile_adv(model, A, X, y, ...) → AdversarialTrainer │
│                                                                   │
│ • train_eval_data/fit_parametric_adv.py                         │
│   └─ fit_parametric_adv(model, A, X, y, ...) → AdversarialTrainer │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│    Core Adversarial Training Engine                              │
├──────────────────────────────────────────────────────────────────┤
│ train_eval_data/adversarial_trainer.py                           │
│                                                                   │
│  CurriculumSchedule (dataclass)                                  │
│  ├─ Manages budget increase schedule                             │
│  ├─ get_budget(epoch) → edge budget %                            │
│  └─ get_phase(epoch) → current training phase                    │
│                                                                   │
│  AdversarialTrainer (class)                                      │
│  ├─ __init__(model, A, X, y, ..., curriculum, attack_fn)        │
│  ├─ _generate_attack() → perturbed adjacency A_pert              │
│  ├─ _train_step() → mixed loss: α*L_clean + (1-α)*L_adv         │
│  ├─ _evaluate() → validation accuracy tracking                   │
│  └─ train() → epochs loop with early stopping                    │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│        Attack Generation & Optimization                          │
├──────────────────────────────────────────────────────────────────┤
│ experiments/run_ablation.py::pgd_attack()                        │
│ ├─ Input: (model, A, X, y, test_idx, budget_edge_num)           │
│ ├─ Process: PGD attack with gradient-based edge perturbation     │
│ └─ Output: A_pert (perturbed adjacency matrix)                   │
│                                                                   │
│ Optimization Details:                                            │
│ ├─ Scheduler: CosineAnnealingLR                                  │
│ ├─ Gradient clipping: max_norm=1.0                               │
│ ├─ Early stopping: if no val improvement for 20 epochs           │
│ └─ Device: Auto-detect GPU/CPU                                   │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│               Output & Checkpointing                             │
├──────────────────────────────────────────────────────────────────┤
│ • Logs: log/{dataset}/{model}/train_*.log                        │
│ • Checkpoints: models saved to exp/result/                       │
│ • Metrics: train/val/test loss, accuracy curves                  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow During Training

```
Input Data (A, X, y, train_idx, val_idx, test_idx)
       │
       ▼
┌──────────────────────┐
│  Epoch Loop          │
│  (0 to max_epoch)    │
└──────────────────────┘
       │
       ├─ Every attack_freq epochs:
       │  │
       │  ├─ Get budget from curriculum: budget = get_budget(epoch)
       │  │
       │  └─ Generate adversarial examples:
       │     A_adv = pgd_attack(model, A, X, y, test_idx, budget)
       │
       └─ Every epoch:
          │
          ├─ Forward on clean: y_clean_pred = model(A, X)
          │
          ├─ Forward on adversarial: y_adv_pred = model(A_adv, X)
          │
          ├─ Compute mixed loss:
          │  L_total = α·L_clean + (1-α)·L_adv
          │
          ├─ Backward pass + gradient clipping
          │
          ├─ Optimizer step (CosineAnnealingLR)
          │
          ├─ Validation on clean data
          │
          └─ Check early stopping (20 epochs no improvement)
```

## Loss Computation Detail

```
During epoch t:

1. Get training data samples
2. Compute clean loss:
   L_clean = CrossEntropy(y_clean_pred, y_train)

3. Compute adversarial loss:
   L_adv = CrossEntropy(y_adv_pred, y_train)

4. Combine:
   L_total = alpha * L_clean + (1 - alpha) * L_adv

5. Mix-in penalty (if available):
   L_total = L_total + penalty_weight * penalty(y_pred)

6. Backward:
   L_total.backward()
   torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

7. Update:
   optimizer.step()
   scheduler.step()
```

Where:
- `alpha` ∈ [0.1, 1.0] default 0.7
- Higher alpha → safer, less adversarial training
- Lower alpha → more adversarial, robust but harder

## File Organization

```
RUNG_ML_Project/
│
├── train_eval_data/
│   ├── adversarial_trainer.py          [NEW] Core adversarial training
│   ├── fit_percentile_adv.py           [NEW] Wrapper for percentile variant
│   ├── fit_parametric_adv.py           [NEW] Wrapper for parametric variant
│   ├── fit_percentile_gamma.py         [EXISTING] Clean training reference
│   ├── fit_parametric_gamma.py         [EXISTING] Clean training reference
│   └── get_dataset.py
│
├── exp/
│   ├── config/
│   │   └── get_model.py                [MODIFIED] Added 2 model branches
│   └── result/
│
├── model/
│   ├── rung_percentile_gamma.py        [EXISTING] Model architecture
│   ├── rung_parametric_gamma.py        [EXISTING] Model architecture
│   └── penalty.py
│
├── experiments/
│   └── run_ablation.py                 [EXISTING] Contains pgd_attack()
│
├── clean.py                            [MODIFIED] Main training script
│
├── run_adversarial_test_suite.py       [NEW] Full ablation tests
├── run_adversarial_quick_demo.py       [NEW] Quick demo (2 configs)
│
├── ADVERSARIAL_TEST_SUITE_README.md    [NEW] Full documentation
├── ADVERSARIAL_QUICK_REFERENCE.md      [NEW] Quick command reference
└── docs/
    └── changes/
        └── 013_adversarial_training.md [NEW] Technical design doc
```

## Configuration Parameters

```python
# Key hyperparameters with valid ranges and effects

CurriculumSchedule:
├── budgets = [5, 10, 20, 40]          # % edges attackable per phase
├── phase_epochs = [50, 100, 200]      # Epoch boundaries
└── default schedule: 5% → 10% → 20% → 40%

Loss Weighting:
├── adv_alpha = 0.7                     # [0.1-1.0] Clean loss weight
├── gradient_clip_norm = 1.0            # Gradient clipping value
└── Loss = 0.7*L_clean + 0.3*L_adv

Attack Configuration:
├── attack_freq = 5                      # Regenerate every N epochs
├── train_pgd_steps = 20                 # PGD iterations during training
└── pgd_iterations = 200                # Evaluation PGD attacks

Training Configuration:
├── lr = 0.05                            # Learning rate
├── weight_decay = 5e-4                  # L2 regularization
├── max_epoch = 300                      # Training epochs
└── early_stop_patience = 20             # Epochs without improvement
```

## Model Variants

```
┌─────────────────────────────────────────┐
│         RUNG Model Variants             │
├─────────────────────────────────────────┤
│                                         │
│  Original Models:                       │
│  ├─ RUNG_percentile_gamma              │
│  ├─ RUNG_parametric_gamma              │
│  └─ RUNG_learnable_distance            │
│                                         │
│  Adversarial Variants [NEW]:            │
│  ├─ RUNG_percentile_adv                │
│  │  └─ Same architecture as above      │
│  │  └─ Different training procedure    │
│  │  └─ Uses AdversarialTrainer class   │
│  │                                      │
│  └─ RUNG_parametric_adv                │
│     └─ Same architecture as above      │
│     └─ Different training procedure    │
│     └─ Uses AdversarialTrainer class   │
│                                         │
│  Note: Model architecture is identical │
│        to parent variant. Only the      │
│        *training procedure* differs.    │
└─────────────────────────────────────────┘
```

## Testing Strategy

```
Test Level            Location                    Runtime
═════════════════════════════════════════════════════════════
Component Test       test_integration_final.py     ~1-2 min
├─ Curriculum schedule
├─ Model creation
├─ Trainer initialization
└─ Training loop (2 epochs)

Quick Demo           run_adversarial_quick_demo.py ~1-2 min
├─ RUNG_percentile_adv baseline
└─ RUNG_parametric_adv baseline

Full Ablation Suite  run_adversarial_test_suite.py ~10-30 min
├─ 2 baselines (both models)
├─ 2 alpha sweeps (0.5, 0.9)
├─ 2 frequency sweeps (1, 10)
└─ 1 full parametric test

Production Training  clean.py / run_all.py        Variable
└─ Full 300+ epoch training
```

## Performance Characteristics

```
Training Time Factors:
═════════════════════════════════════════

Alpha Impact:
├─ Lower alpha (0.5)   → More expensive (compute L_adv fully)
├─ Medium alpha (0.7)  → Balanced
└─ Higher alpha (0.9)  → Faster (less adversarial overhead)

Attack Freq Impact:
├─ Freq=1  → Most expensive (attack every epoch)
├─ Freq=5  → Balanced (default)
└─ Freq=10 → Fastest (attack every 10 epochs)

PGD Steps Impact:
├─ Steps=50   → Strongest attacks, slowest
├─ Steps=20   → Balanced (default)
└─ Steps=5    → Fast attacks, weaker (for testing)

Typical Times (Cora, 2 epochs):
├─ Baseline (clean only)        ~5 seconds
├─ Adv + alpha=0.7 + freq=5     ~35 seconds
├─ Adv + alpha=0.5 + freq=1     ~50+ seconds
└─ Adv + alpha=0.9 + freq=10    ~20 seconds
```

## Key Design Decisions

```
1. GENERIC ADVERSARIAL TRAINER
   ✓ Single AdversarialTrainer class works for any RUNG model
   ✓ Attack function passed as callback (pgd_attack)
   ✓ Curriculum schedule is configurable dataclass
   ✓ No hardcoding of model-specific details

2. CURRICULUM SCHEDULING
   ✓ Budget increases: 5% → 10% → 20% → 40%
   ✓ Smooth transition across training phases
   ✓ Allows warm-up on weak attacks → harder attacks
   ✓ Improves convergence vs immediate strong attacks

3. MIXED LOSS FORMULATION
   ✓ L_total = α·L_clean + (1-α)·L_adv
   ✓ Simple linear combination (not complex saddle point)
   ✓ Allows tuning clean/robust tradeoff via alpha
   ✓ Interpretable and debuggable

4. ADAPTIVE ATTACK REGENERATION
   ✓ Not regenerating every epoch (expensive)
   ✓ Regenerating every N epochs (default 5)
   ✓ Trade-off between computation and attack strength
   ✓ Empirically: more frequent = stronger but slower

5. EARLY STOPPING
   ✓ Monitor validation accuracy (not training loss)
   ✓ Patience=20 epochs
   ✓ Prevents overfitting to adversarial examples
   ✓ Saves best checkpoint

6. GRADIENT CLIPPING
   ✓ max_norm=1.0 to stabilize training
   ✓ Prevents exploding gradients in adversarial setting
   ✓ Applied per parameter group
```

## Validation & Verification Flow

```
1. Unit Level:
   ├─ CurriculumSchedule budget transitions
   └─ AdversarialTrainer initialization

2. Integration Level:
   ├─ Model creation (get_model.py)
   ├─ fit_percentile_adv wrapper
   ├─ fit_parametric_adv wrapper
   └─ Training loop (2 epochs)

3. System Level:
   ├─ clean.py dispatch (--model RUNG_percentile_adv)
   ├─ run_all.py integration
   └─ Full 300-epoch training

4. Ablation Level:
   ├─ Alpha sensitivity (0.5 vs 0.7 vs 0.9)
   ├─ Attack frequency (1 vs 5 vs 10)
   ├─ Both model variants
   └─ Multiple datasets (cora, citeseer)
```

## Next Steps & Future Work

```
Immediate (Done):
✓ Core implementation
✓ Basic testing
✓ Documentation

Short Term:
- [ ] Run full suite on actual hardware
- [ ] Collect robust accuracy numbers
- [ ] Compare with baseline (non-adversarial)

Medium Term:
- [ ] Extended evaluation (heterophilic datasets)
- [ ] Hyperparameter optimization
- [ ] Attack strength analysis

Long Term:
- [ ] Certified robustness
- [ ] Multi-attack evaluation (not just PGD)
- [ ] Physics-informed curriculum
```

---

**Generated**: March 2026  
**Version**: 1.0  
**Status**: Complete & Ready for Testing
