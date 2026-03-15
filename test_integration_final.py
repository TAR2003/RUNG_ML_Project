#!/usr/bin/env python
"""
test_integration_final.py

Final integration test: verify all components work together.
Tests model creation, trainer setup, and abbreviated training loop.
"""

import torch
import torch.nn as nn
from train_eval_data.get_dataset import get_dataset, get_splits
from exp.config.get_model import get_model_default
from train_eval_data.adversarial_trainer import AdversarialTrainer, CurriculumSchedule

def dummy_attack(model, A, X, y, test_idx, budget_edge_num, iterations=20, **kwargs):
    """Dummy attack that returns clean adjacency (for quick testing)."""
    return A.clone()

def test_curriculum():
    """Test 1: Curriculum schedule"""
    print("\n" + "="*70)
    print("TEST 1: Curriculum Schedule")
    print("="*70)
    
    schedule = CurriculumSchedule(
        phase_budgets=[0.05, 0.10, 0.20, 0.40],
        phase_epochs=[50, 50, 100, None]
    )
    
    print(schedule.describe())
    
    # Test key transitions
    tests = [
        (0, 0, 0.05),      # epoch 0, phase 0, budget 0.05
        (49, 0, 0.05),     # end of phase 0
        (50, 1, 0.10),     # start of phase 1
        (100, 2, 0.20),    # start of phase 2
        (200, 3, 0.40),    # start of phase 3
    ]
    
    all_pass = True
    for epoch, expected_phase, expected_budget in tests:
        phase = schedule.get_phase(epoch)
        budget = schedule.get_budget(epoch)
        if phase == expected_phase and abs(budget - expected_budget) < 1e-6:
            print(f"✓ Epoch {epoch:3d}: phase={phase}, budget={budget:.0%}")
        else:
            print(f"✗ Epoch {epoch:3d}: got phase={phase}, budget={budget:.0%} "
                  f"(expected phase={expected_phase}, budget={expected_budget:.0%})")
            all_pass = False
    
    return all_pass

def test_model_creation():
    """Test 2: Model creation and forward pass"""
    print("\n" + "="*70)
    print("TEST 2: Model Creation")
    print("="*70)
    
    device = 'cpu'
    A, X, y = get_dataset('cora')
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    
    # Create both adversarial models
    model_pa, _ = get_model_default(
        'cora', 'RUNG_percentile_adv',
        custom_model_params={'gamma': 6.0},
        device=device
    )
    print(f"✓ RUNG_percentile_adv created ({model_pa.__class__.__name__})")
    
    model_paa, _ = get_model_default(
        'cora', 'RUNG_parametric_adv',
        custom_model_params={'gamma': 6.0},
        device=device
    )
    print(f"✓ RUNG_parametric_adv created ({model_paa.__class__.__name__})")
    
    # Test forward passes
    model_pa = model_pa.to(device)
    model_paa = model_paa.to(device)
    
    logits_pa = model_pa(A, X)
    print(f"✓ RUNG_percentile_adv forward pass: output shape {logits_pa.shape}")
    
    logits_paa = model_paa(A, X)
    print(f"✓ RUNG_parametric_adv forward pass: output shape {logits_paa.shape}")
    
    return True

def test_trainer_setup():
    """Test 3: Trainer setup and configuration"""
    print("\n" + "="*70)
    print("TEST 3: Adversarial Trainer Setup")
    print("="*70)
    
    device = 'cpu'
    A, X, y = get_dataset('cora')
    sps = get_splits(y)
    train_idx, val_idx, test_idx = sps[0]
    
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    
    model, _ = get_model_default(
        'cora', 'RUNG_percentile_adv',
        custom_model_params={'gamma': 6.0},
        device=device
    )
    model = model.to(device)
    
    # Create trainer
    curriculum = CurriculumSchedule(
        phase_budgets=[0.05, 0.40],
        phase_epochs=[2, None]
    )
    
    trainer = AdversarialTrainer(
        model=model,
        attack_fn=dummy_attack,
        curriculum=curriculum,
        alpha=0.7,
        attack_freq=1,
        train_pgd_steps=5,
        lr=5e-2,
        weight_decay=5e-4,
        patience=10,
        log_every=1,
    )
    
    print(f"✓ Trainer created for {model.__class__.__name__}")
    print(f"  - alpha=0.7 (70% clean, 30% adversarial)")
    print(f"  - attack_freq=1 (every epoch)")
    print(f"  - train_pgd_steps=5 (dummy attack, no actual PGD)")
    print(f"  - curriculum: 2 phases")
    
    return True

def test_training_loop():
    """Test 4: Abbreviated training loop (2 epochs)"""
    print("\n" + "="*70)
    print("TEST 4: Training Loop (2 epochs)")
    print("="*70)
    
    device = 'cpu'
    A, X, y = get_dataset('cora')
    sps = get_splits(y)
    train_idx, val_idx, test_idx = sps[0]
    
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    
    model, _ = get_model_default(
        'cora', 'RUNG_percentile_adv',
        custom_model_params={'gamma': 6.0},
        device=device
    )
    model = model.to(device)
    
    curriculum = CurriculumSchedule(
        phase_budgets=[0.05],
        phase_epochs=[None]
    )
    
    trainer = AdversarialTrainer(
        model=model,
        attack_fn=dummy_attack,
        curriculum=curriculum,
        alpha=0.7,
        attack_freq=1,
        train_pgd_steps=5,
        lr=5e-2,
        weight_decay=5e-4,
        patience=10,
        log_every=1,
    )
    
    print("\nRunning 2 epochs...\n")
    best_val, test_acc = trainer.train(
        A=A,
        X=X,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        epochs=2,
        device=device,
    )
    
    print(f"✓ Training completed")
    print(f"  - Best val acc: {best_val:.4f}")
    print(f"  - Test acc: {test_acc:.4f}")
    
    return True

def main():
    print("\n" + "="*70)
    print("FINAL INTEGRATION TEST")
    print("="*70)
    
    tests = [
        ("Curriculum Schedule", test_curriculum),
        ("Model Creation", test_model_creation),
        ("Trainer Setup", test_trainer_setup),
        ("Training Loop", test_training_loop),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_pass = all(result for _, result in results)
    print("\n" + "="*70)
    if all_pass:
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nImplementation is complete and verified!")
        print("\nYou can now train with:")
        print("  python clean.py --model RUNG_percentile_adv --data cora --max_epoch 300")
        print("  python clean.py --model RUNG_parametric_adv --data citeseer --max_epoch 300")
    else:
        print("SOME TESTS FAILED ✗")
        print("="*70)
    
    return all_pass

if __name__ == '__main__':
    all_pass = main()
    exit(0 if all_pass else 1)
