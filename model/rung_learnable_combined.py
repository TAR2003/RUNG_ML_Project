"""
model/rung_learnable_combined.py

RUNG_learnable_combined: cosine distance + learnable gamma constrained to
cosine distance space.

Parent model:
    RUNG_learnable_distance (cosine distance + percentile gamma)

Key change:
    Replace percentile-based gamma with learnable gamma that is specifically
    parameterized for cosine distance range [0, 2].

Two gamma modes:
    - per_layer: K independent gamma parameters
        gamma^(k) = sigmoid(raw_gamma[k]) * 2.0

    - schedule: 2-parameter decay schedule
        gamma^(k) = sigmoid(raw_g0) * 2.0 * sigmoid(raw_decay)^k

In both modes, gamma is constrained to (0, 2), matching cosine distance scale.
"""

import numpy as np
import torch
from torch import nn

from model.mlp import MLP
from model.rung_learnable_gamma import scad_weight_differentiable
from model.rung_learnable_distance import DistanceModule
from utils import add_loops, sym_norm


class CosineLearnableGamma(nn.Module):
    """Learnable gamma parameterization tailored to cosine distance range [0, 2]."""

    def __init__(self, prop_step=10, gamma_mode='per_layer', scad_a=3.7):
        super().__init__()
        assert gamma_mode in ('per_layer', 'schedule'), (
            f"gamma_mode must be 'per_layer' or 'schedule', got '{gamma_mode}'"
        )

        self.prop_step = prop_step
        self.gamma_mode = gamma_mode
        self.scad_a = scad_a

        if gamma_mode == 'per_layer':
            # raw_gamma=0.0 -> gamma=1.0 for all layers
            self.raw_gamma = nn.Parameter(torch.zeros(prop_step))
        else:
            # raw_g0=0.0 -> gamma_0=1.0
            # raw_decay=logit(0.85) -> decay_rate=0.85
            self.raw_g0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            logit_085 = float(np.log(0.85 / (1.0 - 0.85)))
            self.raw_decay = nn.Parameter(torch.tensor(logit_085, dtype=torch.float32))

    def get_gamma(self, layer_idx: int) -> torch.Tensor:
        """Return differentiable gamma for layer k."""
        if self.gamma_mode == 'per_layer':
            return torch.sigmoid(self.raw_gamma[layer_idx]) * 2.0

        gamma_0 = torch.sigmoid(self.raw_g0) * 2.0
        decay_rate = torch.sigmoid(self.raw_decay)
        k = torch.tensor(float(layer_idx), device=self.raw_g0.device)
        return gamma_0 * torch.pow(decay_rate, k)

    def get_all_gammas(self):
        """Return current per-layer gamma values as Python floats."""
        with torch.no_grad():
            return [float(self.get_gamma(k).item()) for k in range(self.prop_step)]

    def get_parameters(self):
        """Return parameters for separate optimizer group."""
        if self.gamma_mode == 'per_layer':
            return [self.raw_gamma]
        return [self.raw_g0, self.raw_decay]

    def log_stats(self):
        """Print learned gamma/lambda schedule."""
        gammas = self.get_all_gammas()
        print(f"\n  CosineLearnableGamma ({self.gamma_mode} mode):")
        print(f"  {'Layer':>5}  {'gamma':>8}  {'lam':>8}  {'transition [lam, a*lam]':>25}")
        print(f"  {'-' * 55}")
        for k, g in enumerate(gammas):
            lam = g / self.scad_a
            print(f"  {k:>5}  {g:>8.4f}  {lam:>8.4f}   [{lam:.3f}, {self.scad_a * lam:.3f}]")
        print()


class RUNG_learnable_combined(nn.Module):
    """RUNG with cosine distance and learnable gamma constrained to (0, 2)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list,
        lam_hat: float = 0.9,
        gamma_mode: str = 'per_layer',
        scad_a: float = 3.7,
        prop_step: int = 10,
        dropout: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()

        assert 0 < lam_hat <= 1, "lam_hat must be in (0, 1]"
        assert scad_a > 1.0, "SCAD shape param a must be > 1"
        assert gamma_mode in ('per_layer', 'schedule'), (
            f"gamma_mode must be 'per_layer' or 'schedule', got {gamma_mode}"
        )

        self.lam_hat = lam_hat
        self.lam = 1.0 / lam_hat - 1.0
        self.prop_layer_num = prop_step
        self.scad_a = scad_a
        self.gamma_mode = gamma_mode
        self.eps = eps

        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # Keep identical distance API as parent model, fixed to cosine mode.
        self.distance = DistanceModule(
            hidden_dim=out_dim,
            mode='cosine',
            proj_dim=32,
        )

        self.gamma_module = CosineLearnableGamma(
            prop_step=prop_step,
            gamma_mode=gamma_mode,
            scad_a=scad_a,
        )

        self._last_gammas = [None] * prop_step
        self._last_y_stats = [None] * prop_step

        self._config = {
            'model': 'RUNG_learnable_combined',
            'in_dim': in_dim,
            'out_dim': out_dim,
            'hidden_dims': hidden_dims,
            'lam_hat': lam_hat,
            'gamma_mode': gamma_mode,
            'scad_a': scad_a,
            'prop_step': prop_step,
            'dropout': dropout,
        }

    def get_gamma_parameters(self):
        return self.gamma_module.get_parameters()

    def get_non_gamma_parameters(self):
        gamma_ids = {id(p) for p in self.gamma_module.get_parameters()}
        return [p for p in self.parameters() if id(p) not in gamma_ids]

    def forward(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        # ---- 1. MLP embedding ----
        F0 = self.mlp(F)

        # ---- 2. Graph preprocessing ----
        A = add_loops(A)
        D = A.sum(-1)
        D_sq = D.sqrt().unsqueeze(-1)
        A_tilde = sym_norm(A)

        F = F0

        # ---- 3. QN-IRLS propagation ----
        for k in range(self.prop_layer_num):
            F_norm = F / D_sq
            y = self.distance(F_norm).detach()  # cosine mode has no distance params

            y_no_diag = y[~torch.eye(y.shape[0], dtype=torch.bool, device=y.device)]
            self._last_y_stats[k] = (
                y_no_diag.mean().item(),
                y_no_diag.std().item(),
                y_no_diag.max().item(),
            )

            gamma_k = self.gamma_module.get_gamma(k)
            lam_k = (gamma_k / self.scad_a).clamp(min=self.eps)
            self._last_gammas[k] = float(gamma_k.item())

            W = scad_weight_differentiable(y, lam_k, a=self.scad_a)

            eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            W = W * (1.0 - eye)
            W = torch.where(torch.isnan(W), torch.ones_like(W), W)

            Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)
            F = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat

        return F

    def get_aggregated_features(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return self.forward(A, X)

    def get_last_gammas(self):
        return list(self._last_gammas)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def log_stats(self) -> None:
        print(f"\n{'=' * 70}")
        print(f"{'RUNG_learnable_combined - Statistics (last fwd pass)':^70}")
        print(f"{'=' * 70}")
        print(f"  gamma_mode: {self.gamma_mode}")
        print("  distance_mode: cosine")
        print(f"  parameters: {self.count_parameters()}")
        print(f"\n  {'Layer':>6}  {'gamma':>9}  {'y_mean':>9}  {'y_std':>9}  {'y_max':>9}")
        print(f"  {'-' * 60}")
        for k, (g, ys) in enumerate(zip(self._last_gammas, self._last_y_stats)):
            if g is not None and ys is not None:
                print(
                    f"  {k:>6}  {g:>9.4f}  {ys[0]:>9.4f}  {ys[1]:>9.4f}  {ys[2]:>9.4f}"
                )
        self.gamma_module.log_stats()
        print(f"{'=' * 70}\n")

    def __repr__(self):
        n_gamma = len(self.gamma_module.get_parameters())
        return (
            "RUNG_learnable_combined(\n"
            f"  gamma_mode={self.gamma_mode} ({n_gamma} gamma param tensor(s)),\n"
            "  distance=cosine,\n"
            f"  prop_step={self.prop_layer_num},\n"
            f"  params={self.count_parameters()}\n"
            ")"
        )
