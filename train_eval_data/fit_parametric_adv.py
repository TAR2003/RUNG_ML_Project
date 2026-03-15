"""
train_eval_data/fit_parametric_adv.py

Training function for RUNG_parametric_gamma with curriculum adversarial training.

Identical to fit_percentile_adv except it wraps RUNG_parametric_gamma instead.
The AdversarialTrainer automatically detects and uses parameter groups
if the model provides get_non_gamma_parameters() and get_gamma_parameters().

Usage:
    from train_eval_data.fit_parametric_adv import fit_parametric_adv
    from experiments.run_ablation import pgd_attack
    
    fit_parametric_adv(
        model, A, X, y, train_idx, val_idx, test_idx,
        attack_fn=pgd_attack,
        alpha=0.7,
    )
"""

import torch
from train_eval_data.adversarial_trainer import AdversarialTrainer, CurriculumSchedule


def fit_parametric_adv(
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
    **kwargs,
) -> None:
    """
    Training loop for RUNG_parametric_gamma with curriculum adversarial training.

    Identical interface to fit_percentile_adv.
    The AdversarialTrainer automatically handles parameter groups for
    gamma_0 and decay_rate if the model provides the helper methods.

    Args:
        model:                RUNG_parametric_gamma instance
        A:                    [N, N] adjacency matrix
        X:                    [N, D] node feature matrix
        y:                    [N] integer class labels
        train_idx:            Training node indices
        val_idx:              Validation node indices
        test_idx:             Test node indices (used for attack margin)
        attack_fn:            Attack function (e.g., pgd_attack from experiments.run_ablation)
        alpha:                Weight on clean loss (0.7 recommended)
        attack_freq:          Regenerate attack every N epochs
        train_pgd_steps:      PGD iterations during training
        lr:                   Learning rate
        weight_decay:         L2 regularization
        max_epoch:            Maximum training epochs
        patience:             Early stopping patience
        log_every:            Log frequency
        curriculum_budgets:   List of attack budgets per phase
        curriculum_epochs:    List of epoch counts per phase
        test_budget:          Budget used at test time (reference)
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
