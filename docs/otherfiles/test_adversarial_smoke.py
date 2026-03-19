#!/usr/bin/env python
"""
test_adversarial_smoke.py

Smoke test for adversarial training with dummy attack.
Just checks that the training loop runs for 3 epochs without errors.
"""

import torch
import torch.nn as nn
from train_eval_data.get_dataset import get_dataset, get_splits
from exp.config.get_model import get_model_default
from train_eval_data.adversarial_trainer import AdversarialTrainer, CurriculumSchedule

def dummy_attack(model, A, X, y, test_idx, budget_edge_num, iterations=20, **kwargs):
    """
    Dummy attack that just returns the clean adjacency.
    Used for testing without running expensive PGD.
    """
    return A.clone()


def main():
    print("=" * 70)
    print("Adversarial Training Smoke Test")
    print("=" * 70)
    
    device = 'cpu'
    A, X, y = get_dataset('cora')
    sps = get_splits(y)
    train_idx, val_idx, test_idx = sps[0]
    
    # Move to device
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    
    # Create a small model for quick testing
    model, _ = get_model_default(
        'cora', 'RUNG_percentile_adv',
        custom_model_params={'gamma': 6.0},
        device=device
    )
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create curriculum (very short for testing)
    curriculum = CurriculumSchedule(
        phase_budgets=[0.05, 0.40],
        phase_epochs=[2, None],  # Only 2 epochs in phase 0
    )
    print(f"\nCurriculum phases: 2")
    print(f"  Phase 0: 2 epochs at budget=5%")
    print(f"  Phase 1: forever at budget=40%")
    
    # Create adversarial trainer
    trainer = AdversarialTrainer(
        model=model,
        attack_fn=dummy_attack,  # Use dummy attack for speed
        curriculum=curriculum,
        alpha=0.7,
        attack_freq=1,  # Attack every epoch
        train_pgd_steps=5,
        lr=5e-2,
        weight_decay=5e-4,
        patience=10,
        log_every=1,
    )
    
    print(f"\nTrainer config:")
    print(f"  alpha=0.7, attack_freq=1, train_pgd_steps=5")
    print(f"  Using dummy attack (returns clean adjacency)")
    
    # Run training for 3 epochs
    print(f"\nRunning training for 3 epochs...")
    best_val, test_acc = trainer.train(
        A=A,
        X=X,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        epochs=3,
        device=device,
    )
    
    print(f"\n" + "=" * 70)
    print(f"SMOKE TEST PASSED")
    print(f"  Best val acc: {best_val:.4f}")
    print(f"  Test acc: {test_acc:.4f}")
    print(f"=" * 70)


if __name__ == '__main__':
    main()
