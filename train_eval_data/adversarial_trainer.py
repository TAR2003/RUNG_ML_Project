"""
train_eval_data/adversarial_trainer.py

Curriculum adversarial training wrapper for RUNG-style models.

Implements mixed clean+adversarial loss training:
    L_total = alpha * L_clean + (1 - alpha) * L_adv

with automatic curriculum budget scheduling and adaptive attack generation.

This is a general-purpose trainer that works with any RUNG model and any
attack function. It handles the training loop while keeping models and
attacks completely separate.

Design principles:
    1. Model architecture unchanged — only training procedure changes
    2. Curriculum: training budget increases gradually over epochs
    3. Mixed loss: alpha*L_clean + (1-alpha)*L_adv
    4. Adaptive attack: regenerated every attack_freq epochs using CURRENT model
    5. Configurable: all hyperparameters exposed as arguments

Usage:
    from train_eval_data.adversarial_trainer import AdversarialTrainer, CurriculumSchedule
    from experiments.run_ablation import pgd_attack
    
    trainer = AdversarialTrainer(
        model=my_model,
        attack_fn=pgd_attack,
        curriculum=CurriculumSchedule(...),
        alpha=0.7,
        attack_freq=5,
    )
    best_val, test_acc = trainer.train(
        A, X, y, train_idx, val_idx, test_idx,
        epochs=1000,
        device='cuda'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Tuple
import tqdm


@dataclass
class CurriculumSchedule:
    """
    Defines how the attack budget increases during training.

    The curriculum prevents instability from jumping directly to high-budget
    attacks. Start easy, get harder gradually.

    Example (default):
        phase_budgets  = [0.05, 0.10, 0.20, 0.40]
        phase_epochs   = [50,   50,   100,  None]

        Epochs 1-50:    budget = 5%
        Epochs 51-100:  budget = 10%
        Epochs 101-200: budget = 20%
        Epochs 201+:    budget = 40%  (None = stay here forever)

    Args:
        phase_budgets: List of attack budgets (fraction of edges)
        phase_epochs:  List of epoch counts per phase (None = infinite)
        test_budget:   Budget used at evaluation time (for logging)
    """
    phase_budgets: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.20, 0.40]
    )
    phase_epochs: List[Optional[int]] = field(
        default_factory=lambda: [50, 50, 100, None]
    )
    test_budget: float = 0.40

    def __post_init__(self):
        assert len(self.phase_budgets) == len(self.phase_epochs), \
            "phase_budgets and phase_epochs must have same length"
        assert self.phase_epochs[-1] is None, \
            "Last phase must have None epoch count (stay there forever)"

    def get_budget(self, epoch: int) -> float:
        """Return the training attack budget for a given epoch (0-indexed)."""
        cumulative = 0
        for budget, n_epochs in zip(self.phase_budgets, self.phase_epochs):
            if n_epochs is None:
                return budget
            cumulative += n_epochs
            if epoch < cumulative:
                return budget
        return self.phase_budgets[-1]

    def get_phase(self, epoch: int) -> int:
        """Return current curriculum phase (0-indexed)."""
        cumulative = 0
        for i, (budget, n_epochs) in enumerate(
                zip(self.phase_budgets, self.phase_epochs)):
            if n_epochs is None:
                return i
            cumulative += n_epochs
            if epoch < cumulative:
                return i
        return len(self.phase_budgets) - 1

    def describe(self) -> str:
        lines = ["Curriculum schedule:"]
        cumulative = 0
        for i, (b, n) in enumerate(zip(self.phase_budgets, self.phase_epochs)):
            if n is None:
                lines.append(f"  Phase {i}: epoch {cumulative}+ → budget={b:.0%}")
            else:
                lines.append(
                    f"  Phase {i}: epoch {cumulative}-{cumulative+n-1} → budget={b:.0%}"
                )
                cumulative += n
        return "\n".join(lines)


class AdversarialTrainer:
    """
    Curriculum adversarial training wrapper for RUNG-style models.

    Works with dense adjacency matrices and index-based masks (NOT PyG Data).

    Implements:
        L_total = alpha * L_clean + (1 - alpha) * L_adv

    Args:
        model:          Any RUNG-style nn.Module with forward(A, X) interface
        attack_fn:      Attack function with signature:
                            attack_fn(model, A, X, y, test_idx, budget_edge_num, iterations)
                            → A_pert (perturbed adjacency matrix)
        curriculum:     CurriculumSchedule instance
        alpha:          Weight on clean loss. Range [0,1].
                        0.7 = 70% clean, 30% adversarial (recommended start)
        attack_freq:    Regenerate attack every N epochs. Higher = faster training.
                        1 = every epoch (strongest), 5 = every 5 epochs (balanced)
        train_pgd_steps: PGD steps during training. Fewer = faster.
                        20-50 is typically enough. Use full steps only at test.
        attack_kwargs:  Extra kwargs passed to attack_fn (e.g., grad_clip)
        lr:             Learning rate
        weight_decay:   L2 regularization
        patience:       Early stopping patience (epochs)
        log_every:      Log frequency (epochs)
    """

    def __init__(
        self,
        model: nn.Module,
        attack_fn: Callable,
        curriculum: Optional[CurriculumSchedule] = None,
        alpha: float = 0.7,
        attack_freq: int = 5,
        train_pgd_steps: int = 20,
        attack_kwargs: Optional[Dict] = None,
        lr: float = 5e-2,
        weight_decay: float = 5e-4,
        patience: int = 100,
        log_every: int = 10,
    ):
        self.model           = model
        self.attack_fn       = attack_fn
        self.curriculum      = curriculum or CurriculumSchedule()
        self.alpha           = alpha
        self.attack_freq     = attack_freq
        self.train_pgd_steps = train_pgd_steps
        self.attack_kwargs   = attack_kwargs or {}
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.patience        = patience
        self.log_every       = log_every

        # Internal state
        self._cached_A_pert = None

    def _build_optimizer(self):
        """
        Build optimizer respecting model's parameter groups if they exist.

        Some models (RUNG_parametric_gamma) have separate parameter groups
        for gamma parameters with different learning rates. This method
        preserves those groups if they exist.
        """
        # Try to use model's own optimizer builder if it exists
        if hasattr(self.model, 'build_optimizer'):
            return self.model.build_optimizer(
                lr=self.lr, weight_decay=self.weight_decay
            )

        # Try to detect parameter groups (RUNG_parametric_gamma style)
        if hasattr(self.model, 'get_gamma_parameters') and \
           hasattr(self.model, 'get_non_gamma_parameters'):
            gamma_lr = self.lr * 0.3  # Gamma parameters get lower LR
            return torch.optim.Adam([
                {
                    'params':       list(self.model.get_non_gamma_parameters()),
                    'lr':           self.lr,
                    'weight_decay': self.weight_decay,
                },
                {
                    'params':       list(self.model.get_gamma_parameters()),
                    'lr':           gamma_lr,
                    'weight_decay': 0.0,
                },
            ])

        # Default: single group
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _generate_attack(self, A, X, y, test_idx, train_budget, device):
        """
        Generate adversarial graph using current model weights (adaptive).

        CRITICAL: Model must be in eval() mode during attack generation.
        This prevents batch norm / dropout from affecting gradient computation
        inside the attack. Switch back to train() after.

        Args:
            A:               [N, N] clean adjacency matrix
            X:               [N, D] node features
            y:               [N] labels
            test_idx:        Indices to use for attack margin
            train_budget:    Attack budget (fraction of edges)
            device:          torch device

        Returns:
            A_pert: [N, N] perturbed adjacency matrix
            attack_time: float, seconds elapsed
        """
        self.model.eval()

        # Move to device
        A = A.to(device)
        X = X.to(device)
        y = y.to(device)
        test_idx = test_idx.to(device)

        t0 = time.time()

        # Compute budget in terms of edge count
        # A is dense [N,N] adjacency, so number of edges = count_nonzero / 2
        n_edges = A.count_nonzero().item() // 2
        budget_edge_num = max(1, int(train_budget * n_edges))

        # Call the attack function
        # pgd_attack expects: model, A, X, y, test_idx, budget_edge_num, iterations
        A_pert = self.attack_fn(
            self.model, A, X, y, test_idx, budget_edge_num,
            iterations=self.train_pgd_steps,
            **self.attack_kwargs
        )

        attack_time = time.time() - t0
        self.model.train()

        return A_pert, attack_time

    def _train_step(self, A, A_pert, X, y, train_idx, optimizer):
        """
        Single gradient update step with mixed clean + adversarial loss.

        Computes:
            L = alpha * CE(model(A, X)[train_idx], y[train_idx])
              + (1-alpha) * CE(model(A_pert, X)[train_idx], y[train_idx])

        Args:
            A:          [N, N] clean adjacency
            A_pert:     [N, N] perturbed adjacency
            X:          [N, D] features
            y:          [N] labels
            train_idx:  Training node indices
            optimizer:  Adam optimizer

        Returns:
            total_loss, clean_loss, adv_loss  (all floats)
        """
        self.model.train()
        optimizer.zero_grad()

        # Clean graph forward
        logits_clean = self.model(A, X)
        L_clean = F.cross_entropy(logits_clean[train_idx], y[train_idx])

        # Adversarial graph forward
        logits_adv = self.model(A_pert, X)
        L_adv = F.cross_entropy(logits_adv[train_idx], y[train_idx])

        # Mixed loss
        L_total = self.alpha * L_clean + (1.0 - self.alpha) * L_adv

        L_total.backward()

        # Gradient clipping — important for stability with adversarial gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        optimizer.step()

        return L_total.item(), L_clean.item(), L_adv.item()

    @torch.no_grad()
    def _evaluate(self, A, X, y, indices, device):
        """Evaluate accuracy on indexed nodes."""
        self.model.eval()
        logits = self.model(A.to(device), X.to(device))
        preds = logits[indices].argmax(dim=-1)
        return (preds == y[indices]).float().mean().item()

    def train(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        test_idx: torch.Tensor,
        epochs: int = 1000,
        device: str = 'cpu',
    ) -> Tuple[float, float]:
        """
        Full adversarial training loop.

        Phase structure:
            1. Curriculum warmup: easy attacks first
            2. Joint training: clean + adversarial loss at each step
            3. Attack regeneration: every attack_freq epochs
            4. Early stopping: on validation accuracy

        Args:
            A:        [N, N] adjacency matrix (dense, no self-loops)
            X:        [N, D] node feature matrix
            y:        [N] integer class labels
            train_idx: Training node indices
            val_idx:   Validation node indices
            test_idx:  Test node indices (used for attack margin)
            epochs:    Maximum training epochs
            device:    'cpu' or 'cuda'

        Returns:
            best_val_acc:  float
            test_acc:      float
        """
        self.model = self.model.to(device)
        A = A.to(device)
        X = X.to(device)
        y = y.to(device)
        train_idx = train_idx.to(device)
        val_idx = val_idx.to(device)
        test_idx = test_idx.to(device)

        optimizer = self._build_optimizer()

        best_val_acc = 0.0
        best_epoch = 0
        best_state = None

        # Print training config
        print(f"\n{'='*68}")
        print(f"{'Adversarial Training':^68}")
        print(f"{'='*68}")
        print(f"  Model:         {self.model.__class__.__name__}")
        print(f"  alpha:         {self.alpha} "
              f"(clean={self.alpha:.0%}, adv={1-self.alpha:.0%})")
        print(f"  attack_freq:   every {self.attack_freq} epochs")
        print(f"  train_pgd_steps: {self.train_pgd_steps}")
        print(f"  patience:      {self.patience}")
        print()
        print(self.curriculum.describe())
        print(f"{'='*68}\n")

        # Initialize attacked data cache
        self._cached_A_pert = A.clone()
        total_attack_time = 0.0

        for epoch in tqdm.trange(epochs, desc="adversarial_train"):

            # Get current curriculum budget
            train_budget = self.curriculum.get_budget(epoch)
            phase = self.curriculum.get_phase(epoch)

            # ---- Generate / refresh attack ----
            if epoch % self.attack_freq == 0 or epoch == 0:
                A_pert, att_time = self._generate_attack(
                    A, X, y, test_idx, train_budget, device
                )
                self._cached_A_pert = A_pert
                total_attack_time += att_time
            else:
                A_pert = self._cached_A_pert

            # ---- Training step ----
            total_loss, clean_loss, adv_loss = self._train_step(
                A, A_pert, X, y, train_idx, optimizer
            )

            # ---- Validation ----
            val_acc = self._evaluate(A, X, y, val_idx, device)

            # ---- Track best ----
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

            # ---- Logging ----
            if (epoch + 1) % self.log_every == 0:
                print(
                    f"Ep {epoch+1:4d} | phase={phase} bgt={train_budget:.0%} | "
                    f"L={total_loss:.4f} "
                    f"(Lc={clean_loss:.4f} La={adv_loss:.4f}) | "
                    f"val={val_acc:.4f}"
                )

            # ---- Early stopping ----
            if epoch - best_epoch > self.patience:
                print(
                    f"\nEarly stopping at epoch {epoch + 1}. "
                    f"Best val={best_val_acc:.4f} at epoch {best_epoch + 1}"
                )
                break

        # ---- Restore best model ----
        if best_state is not None:
            self.model.load_state_dict(best_state)

        test_acc = self._evaluate(A, X, y, test_idx, device)

        print(f"\n{'='*68}")
        print(f"Training complete")
        print(f"  Best val:  {best_val_acc:.4f} (epoch {best_epoch + 1})")
        print(f"  Test acc:  {test_acc:.4f}")
        print(f"  Total attack time: {total_attack_time:.1f}s")
        print(f"{'='*68}\n")

        return best_val_acc, test_acc
