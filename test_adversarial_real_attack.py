#!/usr/bin/env python
"""
test_adversarial_real_attack.py

Test adversarial training with real pgd_attack function.
Runs for 2 epochs with low PGD steps to keep it fast.
"""

import torch
from train_eval_data.get_dataset import get_dataset, get_splits
from exp.config.get_model import get_model_default
from train_eval_data.fit_percentile_adv import fit_percentile_adv
from experiments.run_ablation import pgd_attack

def main():
    print("=" * 70)
    print("Adversarial Training with Real Attack")
    print("=" * 70)
    
    device = torch.device('cpu')
    A, X, y = get_dataset('cora')
    sps = get_splits(y)
    train_idx, val_idx, test_idx = sps[0]
    
    # Create model
    model, _ = get_model_default(
        'cora', 'RUNG_percentile_adv',
        custom_model_params={'gamma': 6.0},
        device=device
    )
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Dataset: cora")
    print(f"  Nodes: {A.shape[0]}, Edges: {A.count_nonzero().item() // 2}")
    
    # Train with adversarial training (very short)
    print(f"\nRunning adversarial training with pgd_attack...")
    print(f"  max_epoch=2, attack_freq=1, train_pgd_steps=5 (VERY SHORT FOR TESTING)")
    print(f"  budget=0.05 (very small for speed)")
    print(f"  alpha=0.7")
    
    fit_percentile_adv(
        model,
        A, X, y, train_idx, val_idx, test_idx,
        attack_fn=pgd_attack,
        alpha=0.7,
        attack_freq=2,  # Attack every 2 epochs
        train_pgd_steps=5,  # Only 5 PGD steps (very fast)
        lr=5e-2,
        weight_decay=5e-4,
        max_epoch=2,
        patience=10,
        log_every=1,
        curriculum_budgets=[0.05],  # Single budget (no curriculum phases)
        curriculum_epochs=[None],   # Stay at 5% forever
    )
    
    print(f"\n" + "=" * 70)
    print(f"REAL ATTACK TEST PASSED")
    print(f"=" * 70)


if __name__ == '__main__':
    main()
