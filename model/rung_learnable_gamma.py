"""
model/rung_learnable_gamma.py

Extends RUNG_new_SCAD by replacing the fixed scalar SCAD threshold (gamma)
with per-layer learnable parameters trained end-to-end via backpropagation.

Model lineage:
    RUNG (base, NeurIPS 2024)
        └── RUNG_new_SCAD  (replaces MCP with SCAD penalty, fixed gamma)
                └── RUNG_learnable_gamma  (THIS FILE — per-layer learnable gamma)

Key change:
    Each of the K QN-IRLS aggregation layers has its own SCAD threshold
        lam^(k) = exp(log_lam^(k))
    where log_lam^(k) is an nn.Parameter updated by the optimizer.

Mathematical motivation (see docs/changes/007_learnable_gamma.md):
    Feature differences y_ij = ||f_i/√d_i - f_j/√d_j||_2 shrink across
    aggregation layers as features become smoother.  A single fixed gamma is
    suboptimal at every layer depth.  Per-layer gamma lets each layer find its
    own optimal SCAD threshold through gradient descent.

Naming convention:
    The CLI / logging parameter `gamma` refers to the "zero-cutoff" of SCAD:
        zero-cutoff = a * lam  (region 3 starts at y >= a*lam)
    Internally we store log_lam (lam = zero-cutoff / a).
    With default a=3.7:
        lam = gamma / 3.7  (matches RUNG_new_SCAD convention exactly)

Interface:
    forward(A, X)  — same signature as RUNG, A is dense [N,N] adjacency,
                     X is [N, D] node feature matrix.

Do NOT modify RUNG or RUNG_new_SCAD in any way.

Reference: "Robust Graph Neural Networks via Unbiased Aggregation" NeurIPS 2024
"""

import numpy as np
import torch
from torch import nn

from model.mlp import MLP
from utils import add_loops, pairwise_squared_euclidean, sym_norm


# ============================================================
#  DIFFERENTIABLE SCAD WEIGHT FUNCTION
# ============================================================

def scad_weight_differentiable(
    y: torch.Tensor,
    lam: torch.Tensor,
    a: float = 3.7,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    SCAD penalty derivative for IRLS edge reweighting.

    FULLY DIFFERENTIABLE w.r.t. `lam` via torch.where (no boolean indexing).
    This is required for gradients to flow from the loss into each layer's
    learnable log_lam parameter.

    dρ_SCAD(y)/dy² =
        1/(2·y)                              if y < lam          [region 1]
        (a·lam - y) / ((a-1)·lam · 2·y)    if lam ≤ y < a·lam  [region 2]
        0                                    if y ≥ a·lam        [region 3]

    Args:
        y:   Pairwise feature diffs, shape [N, N], values ≥ 0.
        lam: SCAD threshold, scalar tensor with requires_grad=True (learnable).
        a:   SCAD shape parameter, default 3.7 (Fan & Li 2001).
        eps: Numerical stability floor for division.

    Returns:
        W:   Edge weights, shape [N, N], values ≥ 0.

    Gradient notes:
        dW/d(lam) is non-zero ONLY for edges in region 2 (lam ≤ y < a·lam).
        Gradient flows only through "transition-zone" edges, which is exactly
        the information needed to calibrate the threshold per layer.

    Important: uses torch.where throughout — do NOT refactor to boolean
    indexing (W[mask] = ...) as that would break autograd w.r.t. lam.
    """
    y_safe = y.clamp(min=eps)

    # Compute all three region values unconditionally.
    # torch.where selects between them; both branches must be computed
    # but gradients only flow through the selected branch.
    region1_val = 1.0 / (2.0 * y_safe)

    # Region 2: linearly decaying weight; clamp denominator for safety
    denom2 = ((a - 1.0) * lam * 2.0 * y_safe).clamp(min=eps)
    region2_val = (a * lam - y) / denom2
    region2_val = region2_val.clamp(min=0.0)   # safety: no negative weights

    region3_val = torch.zeros_like(y)

    # Region boundaries (use original y, not y_safe, to match math)
    in_region1 = y < lam
    in_region2 = (y >= lam) & (y < a * lam)
    # in_region3: y >= a*lam  (implicit: torch.where default)

    # Compose with torch.where — preserves autograd for lam
    W = torch.where(
        in_region1,
        region1_val,
        torch.where(in_region2, region2_val, region3_val),
    )
    return W


# ============================================================
#  FULL MODEL: RUNG_learnable_gamma
# ============================================================

class RUNG_learnable_gamma(nn.Module):
    """
    RUNG with per-layer learnable SCAD threshold (gamma) parameters.

    Architecture (identical to RUNG_new_SCAD except gamma is learnable):
        1. MLP: X → F⁰   (maps raw features to class-space embeddings)
        2. K QN-IRLS graph aggregation layers, each with its own lam^(k)
        3. Returns F^(K) as logits (no separate classification head)

    The K learnable log_lam parameters are registered via nn.ParameterList
    and automatically included in model.parameters() — the optimizer updates
    them alongside all MLP weights.

    For a two-group optimizer (different LR for gamma vs other params), use
    the helper methods get_gamma_parameters() / get_non_gamma_parameters().

    Args:
        in_dim:               Input feature dimension D.
        out_dim:              Number of output classes C.
        hidden_dims:          MLP hidden layer widths, e.g. [64].
        lam_hat:              Skip-connection fraction λ̂ ∈ (0, 1].
                              λ = 1/λ̂ − 1  (larger λ̂ → stronger smoothing).
        gamma_init:           Initial "zero-cutoff" γ for all layers.
                              Internally: lam_init = gamma_init / a.
                              Matches the CLI --gamma convention of RUNG_new_SCAD.
        gamma_init_strategy:  How to stagger initial gammas across layers:
                              'uniform'   — all layers at gamma_init (default)
                              'decreasing'— gamma_init * exp(-k/(K-1))
                              'increasing'— gamma_init * exp(+k/(K-1))
        scad_a:               SCAD shape parameter a, default 3.7.
        prop_step:            Number of QN-IRLS iterations K, default 10.
        dropout:              MLP dropout rate.

    Parameter count vs RUNG_new_SCAD:
        Added: K scalar parameters (one log_lam per layer).
        For K=10: only 10 extra parameters — negligible vs MLP weights.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list,
        lam_hat: float = 0.9,
        gamma_init: float = 6.0,
        gamma_init_strategy: str = "uniform",
        scad_a: float = 3.7,
        prop_step: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ---- Validate ----
        assert 0 < lam_hat <= 1, "lam_hat must be in (0, 1]"
        assert gamma_init > 0,   "gamma_init must be positive"
        assert scad_a > 1.0,     "SCAD shape param a must be > 1"

        # ---- Hyperparams ----
        self.lam_hat       = lam_hat
        self.lam           = 1.0 / lam_hat - 1.0   # skip-conn weight λ
        self.prop_layer_num = prop_step
        self.scad_a        = scad_a
        self.gamma_init    = gamma_init
        self.gamma_init_strategy = gamma_init_strategy

        # ---- MLP backbone (same as RUNG) ----
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # ---- Per-layer learnable SCAD thresholds ----
        # We learn log_lam^(k) and compute lam^(k) = exp(log_lam^(k)).
        # Initialization: lam_init = gamma_init / a  (matching RUNG_new_SCAD
        # where lam = gamma / a for fair comparison at equivalent gamma).
        #
        # Using log-space parameterization:
        #   1. Guarantees lam > 0 always (exp is always positive)
        #   2. Log-space gradients = relative changes → more numerically stable
        #   3. Smooth gradient landscape even for large lam values
        gamma_zeros = self._init_gammas(gamma_init, prop_step, gamma_init_strategy)
        lam_inits   = [g / scad_a for g in gamma_zeros]  # lam = gamma / a

        self.log_lams = nn.ParameterList([
            nn.Parameter(
                torch.tensor(float(np.log(l)), dtype=torch.float32)
            )
            for l in lam_inits
        ])

        # ---- Save config for logging / checkpointing ----
        self._config = {
            "model":                 "RUNG_learnable_gamma",
            "in_dim":                in_dim,
            "out_dim":               out_dim,
            "hidden_dims":           hidden_dims,
            "lam_hat":               lam_hat,
            "gamma_init":            gamma_init,
            "gamma_init_strategy":   gamma_init_strategy,
            "scad_a":                scad_a,
            "prop_step":             prop_step,
            "dropout":               dropout,
        }

    # ------------------------------------------------------------------
    #  Gamma initialisation strategy
    # ------------------------------------------------------------------

    @staticmethod
    def _init_gammas(
        gamma_init: float,
        num_layers: int,
        strategy: str,
    ) -> list:
        """
        Return list of initial gamma (zero-cutoff) values, one per layer.

        Strategies:
            'uniform':    all layers at gamma_init.
            'decreasing': gamma_k = gamma_init * exp(-k / (K-1))
                          Motivated by feature diffs shrinking with depth —
                          later layers should prune at smaller thresholds.
            'increasing': gamma_k = gamma_init * exp(+k / (K-1))
                          Alternative: earlier layers need finer thresholds.
        """
        K = num_layers
        if strategy == "uniform":
            return [gamma_init] * K
        elif strategy == "decreasing":
            return [
                gamma_init * float(np.exp(-k / max(K - 1, 1)))
                for k in range(K)
            ]
        elif strategy == "increasing":
            return [
                gamma_init * float(np.exp(k / max(K - 1, 1)))
                for k in range(K)
            ]
        else:
            raise ValueError(
                f"Unknown gamma_init_strategy: '{strategy}'. "
                f"Choose from: uniform, decreasing, increasing."
            )

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: MLP embedding → K QN-IRLS layers with learnable gamma.

        Same algorithm as RUNG.forward (quasi_newton=True) except each layer k
        uses its own learnable SCAD threshold lam^(k) = exp(log_lam^(k)).

        Args:
            A: [N, N] adjacency matrix (float, no self-loops expected).
            F: [N, D] node feature matrix.

        Returns:
            F: [N, C] logit matrix after K propagation iterations.
        """
        # ---- 1. MLP: raw features → initial class-space features F⁰ ----
        F0 = self.mlp(F)

        # ---- 2. Preprocessing (identical to RUNG) ----
        A       = add_loops(A)                    # add self-loop (avoids zero degree)
        D       = A.sum(-1)                       # [N] degree vector
        D_sq    = D.sqrt().unsqueeze(-1)          # [N, 1]  for degree-norm
        A_tilde = sym_norm(A)                     # D^{-1/2} A D^{-1/2}

        F = F0

        # ---- 3. QN-IRLS propagation — per-layer learnable gamma ----
        for k, log_lam_k in enumerate(self.log_lams):

            # Compute learnable SCAD threshold for layer k (differentiable)
            lam_k = torch.exp(log_lam_k)          # scalar tensor, grad flows here

            # y_{ij} = ||f_i/√d_i - f_j/√d_j||_2  (Eq. 8)
            # .detach() on Z matches RUNG convention: IRLS treats y as fixed
            # in each outer iteration (only the linear solve is differentiated).
            Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
            y = Z.sqrt()

            # W_{ij} = dρ_SCAD(y_ij)/dy²  with learnable lam_k
            # Uses torch.where — gradient flows through lam_k (not broken by masking)
            W = scad_weight_differentiable(y, lam_k, a=self.scad_a)

            # Zero diagonal — use out-of-place multiplication so that the
            # computation graph for lam_k is NOT broken.
            # W[idx, idx] = 0.0  would be an in-place op that breaks autograd.
            eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            W   = W * (1.0 - eye)

            # NaN guard (isolated nodes may produce 0/0) — out-of-place
            W = torch.where(torch.isnan(W), torch.ones_like(W), W)

            # QN-IRLS update (Eq. 8 in paper):
            #   F^(k+1) = (diag(q) + λI)^{-1} [(W ⊙ Ã) F^(k) + λ F^(0)]
            # diag(q)_i = Σ_j W_ij A_ij / d_i
            Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)   # [N, 1]
            F     = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat

        return F

    # ------------------------------------------------------------------
    #  Feature extraction (estimation bias compatibility)
    # ------------------------------------------------------------------

    def get_aggregated_features(
        self, A: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract propagated features after all QN-IRLS layers.

        Since RUNG_learnable_gamma propagates in class space, the aggregated
        features ARE the final logits.  Provided for compatibility with the
        estimation bias computation in utils/metrics.py.

        Args:
            A: [N, N] adjacency matrix.
            X: [N, D] node feature matrix.

        Returns:
            F: [N, C] aggregated feature matrix.
        """
        return self.forward(A, X)

    # ------------------------------------------------------------------
    #  Optimizer helpers
    # ------------------------------------------------------------------

    def get_gamma_parameters(self):
        """
        Yield only the log_lam parameters (one per layer).

        Use for a separate optimizer parameter group to apply a different
        learning rate to gamma:

            optimizer = torch.optim.Adam([
                {'params': list(model.get_non_gamma_parameters()), 'lr': 0.01},
                {'params': list(model.get_gamma_parameters()),     'lr': 0.002},
            ])
        """
        return (p for p in self.log_lams.parameters())

    def get_non_gamma_parameters(self):
        """Yield all parameters EXCEPT the log_lam (gamma) parameters."""
        gamma_ids = {id(p) for p in self.log_lams.parameters()}
        return (p for p in self.parameters() if id(p) not in gamma_ids)

    # ------------------------------------------------------------------
    #  Inspection / logging
    # ------------------------------------------------------------------

    def get_learned_gammas(self) -> list:
        """
        Return current learned zero-cutoff (= a * lam) values for all layers.

        Returned values are in the same scale as the CLI --gamma argument
        and as the fixed gamma used in RUNG_new_SCAD, allowing direct comparison.

        Returns:
            gammas: list of float, length num_layers.
        """
        return [
            self.scad_a * float(torch.exp(log_lam).item())
            for log_lam in self.log_lams
        ]

    def get_learned_lams(self) -> list:
        """
        Return current learned lam (SCAD threshold) values for all layers.

        lam^(k) = exp(log_lam^(k))
        zero_cutoff^(k) = a * lam^(k)

        Returns:
            lams: list of float, length num_layers.
        """
        return [float(torch.exp(log_lam).item()) for log_lam in self.log_lams]

    def log_gamma_stats(self) -> None:
        """
        Print a formatted table of learned gamma / lam values per layer.

        Example output:
            ==============================================
              RUNG_learnable_gamma — Gamma Values
            ==============================================
             Layer    lam      gamma(=a·lam)  log_lam
            ----------------------------------------------
                 0    1.6216       6.0000       0.4839
                 1    1.4500       5.3650       0.3716
                ...
        """
        print(f"\n{'=' * 50}")
        print(f"{'RUNG_learnable_gamma — Gamma Values':^50}")
        print(f"{'=' * 50}")
        print(f"{'Layer':>7}  {'lam':>8}  {'gamma(=a*lam)':>14}  {'log_lam':>9}")
        print(f"{'-' * 50}")
        for k, log_lam_param in enumerate(self.log_lams):
            ll  = log_lam_param.item()
            lam = float(np.exp(ll))
            g   = self.scad_a * lam
            print(f"  {k:>5}    {lam:>8.4f}    {g:>14.4f}    {ll:>9.4f}")
        print(f"{'=' * 50}\n")

    def __repr__(self) -> str:
        gammas = self.get_learned_gammas()
        return (
            f"RUNG_learnable_gamma(\n"
            f"  prop_step={self.prop_layer_num},\n"
            f"  lam_hat={self.lam_hat},\n"
            f"  gamma_init={self.gamma_init},\n"
            f"  gamma_init_strategy='{self.gamma_init_strategy}',\n"
            f"  gammas_current=[{', '.join(f'{g:.3f}' for g in gammas)}],\n"
            f"  scad_a={self.scad_a}\n"
            f")"
        )
