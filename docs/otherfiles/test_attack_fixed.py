#!/usr/bin/env python
"""
Quick demo that attack_pgd works with RUNG_combined
Without GPU overhead - just validates the fixed API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from train_eval_data.get_dataset import get_dataset, get_splits
from exp.config.get_model import get_model_default
from train_test_combined import attack_pgd
from utils import accuracy

print("="*80)
print("RUNG_COMBINED + FIXED ATTACK DEMO")
print("="*80)

print("\nLoading Cora dataset...")
A, X, y = get_dataset('cora')
_, _, test_idx = get_splits(y)[0]

print(f"Graph: {A.shape}, features: {X.shape}, classes: {y.unique().shape[0]}")

print("\nCreating RUNG_combined model...")
model_params = {'percentile_q': 0.75, 'use_layerwise_q': False}
model, _ = get_model_default(
    'cora',
    'RUNG_combined',
    custom_model_params=model_params,
    device=torch.device('cpu'),  # Use CPU for demo
)

print(f"Model: {model}\n")

print("\nDoing quick 5-epoch training...")
A = A.to('cpu')
X = X.to('cpu')
y = y.to('cpu')
model = model.to('cpu')

import torch.nn.functional as F
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
train_idx, val_idx, _ = get_splits(y)[0]

for epoch in range(1, 6):
    model.train()
    optimizer.zero_grad()
    logits = model(A, X)
    loss = F.cross_entropy(logits[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        train_acc = accuracy(logits[train_idx], y[train_idx])
    print(f"  Epoch {epoch}: loss={loss.item():.4f}, train_acc={train_acc:.4f}")

print("\n" + "="*80)
print("TESTING ATTACK_PGD FUNCTION")
print("="*80)

# Create a trained model (good for attack)
model.eval()

print("\n[Attack Test 1] Budget 0.05 (5% of edges)")
try:
    acc = attack_pgd(
        model=model,
        A=A,
        X=X,
        y=y,
        test_idx=test_idx,
        budget=0.05,
        n_epochs=2,  # Quick attack
        lr_attack=0.01,
        device='cpu'
    )
    print(f"  ✓ Attack completed!")
    print(f"    - Attacked accuracy: {acc:.4f}")
except Exception as e:
    print(f"  ✗ Attack failed: {e}")
    sys.exit(1)

print("\n[Attack Test 2] Budget 0.20 (20% of edges)")
try:
    acc = attack_pgd(
        model=model,
        A=A,
        X=X,
        y=y,
        test_idx=test_idx,
        budget=0.20,
        n_epochs=2,
        lr_attack=0.01,
        device='cpu'
    )
    print(f"  ✓ Attack completed!")
    print(f"    - Attacked accuracy: {acc:.4f}")
except Exception as e:
    print(f"  ✗ Attack failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✓ SUCCESS - ATTACK FIX WORKS!")
print("="*80)
print("\nThe attack_pgd function is now fixed and ready for full experiments.")
print("Run: python train_test_combined.py --dataset cora --max_epoch 300")
