"""
train_eval_data/fit_percentile_adv.py

Training function for RUNG_percentile_gamma with curriculum adversarial training.

This is a wrapper that:
    1. Imports RUNG_percentile_gamma (architecture unchanged)
    2. Uses AdversarialTrainer to add adversarial training
    3. Delegates to AdversarialTrainer.train()

No new parameters are needed except those controlling adversarial training:
    - alpha: weight on clean loss
    - attack_freq: regenerate attack every N epochs
    - train_pgd_steps: PGD iterations during training

The model architecture is identical to RUNG_percentile_gamma.
Only the training procedure changes.

Usage:
    from train_eval_data.fit_percentile_adv import fit_percentile_adv
    from experiments.run_ablation import pgd_attack
    
    fit_percentile_adv(
        model, A, X, y, train_idx, val_idx, test_idx,
        attack_fn=pgd_attack,
        alpha=0.7,
        attack_freq=5,
        train_pgd_steps=20,
    )
"""

import torch
from train_eval_data.adversarial_trainer import AdversarialTrainer, CurriculumSchedule


def fit_percentile_adv(
    model,
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    attack_fn,
    alpha: float = 0.7,
    attack_freq: int = 5,
    train_pgd_steps: int = 20,
    lr: float = 5e-2,
    weight_decay: float = 5e-4,
    max_epoch: int = 300,
    patience: int = 100,
    log_every: int = 10,
    curriculum_budgets=None,
    curriculum_epochs=None,
    test_budget: float = 0.40,
    attack_kwargs=None,
    **kwargs,  # absorb any extra kwargs for drop-in replacement
) -> None:
    """
    Training loop for RUNG_percentile_gamma with curriculum adversarial training.

    Args:
        model:                RUNG_percentile_gamma instance
        A:                    [N, N] adjacency matrix
        X:                    [N, D] node feature matrix
        y:                    [N] integer class labels
        train_idx:            Training node indices
        val_idx:              Validation node indices
        test_idx:             Test node indices (used for attack margin)
        attack_fn:            Attack function (e.g., pgd_attack from experiments.run_ablation)
        alpha:                Weight on clean loss (0.7 = 70% clean, 30% adv)
        attack_freq:          Regenerate attack every N epochs
        train_pgd_steps:      PGD iterations during training (typically 20-50)
        lr:                   Learning rate for model parameters
        weight_decay:         L2 regularization
        max_epoch:            Maximum training epochs
        patience:             Early stopping patience (epochs)
        log_every:            Log frequency (epochs)
        curriculum_budgets:   List of attack budgets per curriculum phase
        curriculum_epochs:    List of epoch counts per phase
        test_budget:          Budget used at test time (for reference)
        attack_kwargs:        Extra kwargs for attack_fn
        **kwargs:             Ignored; allows drop-in replacement of fit()
    """
    # Default curriculum if not specified
    if curriculum_budgets is None:
        curriculum_budgets = [0.05, 0.10, 0.20, test_budget]
    if curriculum_epochs is None:
        curriculum_epochs = [50, 50, 100, None]

    # Build curriculum schedule
    curriculum = CurriculumSchedule(
        phase_budgets=curriculum_budgets,
        phase_epochs=curriculum_epochs,
        test_budget=test_budget,
    )

    # Create trainer
    trainer = AdversarialTrainer(
        model=model,
        attack_fn=attack_fn,
        curriculum=curriculum,
        alpha=alpha,
        attack_freq=attack_freq,
        train_pgd_steps=train_pgd_steps,
        attack_kwargs=attack_kwargs or {},
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        log_every=log_every,
    )

    # Get device from model
    device = next(model.parameters()).device

    # Run training
    trainer.train(
        A=A,
        X=X,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        epochs=max_epoch,
        device=device.type if hasattr(device, 'type') else str(device),
    )
