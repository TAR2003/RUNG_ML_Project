"""
model/rung_parametric_gamma.py

Extends RUNG_learnable_gamma by replacing K separate gamma parameters
with a 2-parameter exponential decay schedule:

    gamma^(k) = gamma_0 * decay_rate^k

    where:
        gamma_0    = exp(log_gamma_0)      [learnable, > 0]
        decay_rate = sigmoid(raw_decay)    [learnable, in (0,1)]

Model lineage:
    RUNG (NeurIPS 2024)
        └── RUNG_new_SCAD  (SCAD penalty, fixed gamma)
                └── RUNG_learnable_gamma  (per-layer learnable gamma, K parameters)
                        └── RUNG_parametric_gamma  (THIS FILE, 2-parameter schedule)

Why 2 parameters beat K separate parameters:
    - All K layers contribute gradient to both parameters simultaneously
    - Much stronger, more stable gradient signal across layers
    - Forces geometric decrease → encodes depth-smoothing hypothesis explicitly
    - Reduces variance across random seeds dramatically
    - K-2 fewer parameters (e.g., 8 fewer for K=10)

Mathematical motivation:
    Feature differences shrink across aggregation layers as features become smoother.
    With K separate gammas: each layer receives gradient only from its own edges,
    leading to high variance and unstable training (especially if layer k has few
    edges in the SCAD transition zone).
    
    With 2 shared parameters: ALL K layers contribute gradient to both parameters,
    yielding a strong, stable signal. The geometric schedule also encodes the
    hypothesis that later layers need smaller thresholds.

Naming convention (same as RUNG_learnable_gamma):
    The CLI / logging parameter `gamma` refers to the SCAD "zero-cutoff":
        zero-cutoff = a * lam
    Internally we store log_lam (lam = zero-cutoff / a).
    With default a=3.7:
        lam = gamma / 3.7  (matches RUNG_new_SCAD convention exactly)
    
    In this model, gamma_0 is the zero-cutoff at layer 0, so:
        lam^(k) = (gamma_0 / a) * decay_rate^k

Interface:
    forward(A, X)  — identical signature to RUNG_learnable_gamma.
                     A is dense [N,N] adjacency, X is [N, D] node features.

Do NOT modify RUNG, RUNG_new_SCAD, or RUNG_learnable_gamma in any way.

Reference: "Robust Graph Neural Networks via Unbiased Aggregation" NeurIPS 2024
"""

import numpy as np
import torch
from torch import nn

from model.mlp import MLP
from model.rung_learnable_gamma import scad_weight_differentiable
from utils import add_loops, pairwise_squared_euclidean, sym_norm


# ============================================================
#  FULL MODEL: RUNG_parametric_gamma
# ============================================================

class RUNG_parametric_gamma(nn.Module):
    """
    RUNG with 2-parameter exponential gamma decay schedule.

    Instead of K learnable log_lam parameters (one per layer), this model
    learns just 2 scalars:
        log_gamma_0:  controls the initial threshold at layer 0
        raw_decay:    controls the decay rate across layers

    The per-layer threshold is computed as:
        gamma^(k) = gamma_0 * decay_rate^k
        lam^(k) = gamma^(k) / a

    All K layers contribute gradient to both of these 2 parameters,
    resulting in more stable training and fewer parameters.

    Architecture (identical to RUNG_learnable_gamma except for gamma parameterization):
        1. MLP: X → F⁰   (maps raw features to class-space embeddings)
        2. K QN-IRLS graph aggregation layers, each using gamma^(k) from the schedule
        3. Returns F^(K) as logits (no separate classification head)

    For a two-group optimizer (different LR for schedule params vs other params), use
    the helper methods get_gamma_parameters() / get_non_gamma_parameters().

    Args:
        in_dim:               Input feature dimension D.
        out_dim:              Number of output classes C.
        hidden_dims:          MLP hidden layer widths, e.g. [64].
        lam_hat:              Skip-connection fraction λ̂ ∈ (0, 1].
                              λ = 1/λ̂ − 1  (larger λ̂ → stronger smoothing).
        gamma_0_init:         Initial zero-cutoff γ at layer 0 (default 3.0).
        decay_rate_init:      Initial decay rate in (0,1), default 0.85.
                              0.85 means gamma shrinks by 15% per layer.
        scad_a:               SCAD shape parameter a, default 3.7.
        prop_step:            Number of QN-IRLS iterations K, default 10.
        dropout:              MLP dropout rate.

    Parameter count vs RUNG_learnable_gamma:
        Removed: K log_lam scalars
        Added:   2 scalars (log_gamma_0, raw_decay)
        Net:     K-2 fewer parameters (e.g., 8 fewer for K=10)

    Example usage:
        model = RUNG_parametric_gamma(
            in_dim=1433, out_dim=7, hidden_dims=[64],
            lam_hat=0.9, gamma_0_init=3.0, decay_rate_init=0.85
        )
        logits = model(A, X)  # A: [N,N], X: [N,1433] → logits: [N,7]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list,
        lam_hat: float = 0.9,
        gamma_0_init: float = 3.0,
        decay_rate_init: float = 0.85,
        scad_a: float = 3.7,
        prop_step: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ---- Validate ----
        assert 0 < lam_hat <= 1, "lam_hat must be in (0, 1]"
        assert gamma_0_init > 0,   "gamma_0_init must be positive"
        assert 0 < decay_rate_init < 1, "decay_rate_init must be in (0, 1)"
        assert scad_a > 1.0,     "SCAD shape param a must be > 1"

        # ---- Hyperparams ----
        self.lam_hat       = lam_hat
        self.lam           = 1.0 / lam_hat - 1.0   # skip-conn weight λ
        self.prop_layer_num = prop_step
        self.scad_a        = scad_a
        self.gamma_0_init  = gamma_0_init
        self.decay_rate_init = decay_rate_init

        # ---- MLP backbone (same as RUNG_learnable_gamma) ----
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # ---- THE 2 NEW PARAMETERS: Exponential decay schedule ----
        # gamma_0 = exp(log_gamma_0)  ensures gamma_0 > 0 always
        self.log_gamma_0 = nn.Parameter(
            torch.tensor(float(np.log(gamma_0_init)), dtype=torch.float32)
        )

        # decay_rate = sigmoid(raw_decay)  ensures decay_rate in (0,1) always
        # logit(x) = log(x / (1-x))
        logit_init = float(np.log(decay_rate_init / (1.0 - decay_rate_init)))
        self.raw_decay = nn.Parameter(
            torch.tensor(logit_init, dtype=torch.float32)
        )

        # ---- Save config for logging / checkpointing ----
        self._config = {
            "model":              "RUNG_parametric_gamma",
            "in_dim":             in_dim,
            "out_dim":            out_dim,
            "hidden_dims":        hidden_dims,
            "lam_hat":            lam_hat,
            "gamma_0_init":       gamma_0_init,
            "decay_rate_init":    decay_rate_init,
            "scad_a":             scad_a,
            "prop_step":          prop_step,
            "dropout":            dropout,
        }

        # Store last-computed gammas for inspection
        self._last_gammas = [None] * prop_step

    # ------------------------------------------------------------------
    #  Gamma schedule computation
    # ------------------------------------------------------------------

    def get_gamma_0(self) -> torch.Tensor:
        """Return gamma_0 as a differentiable tensor."""
        return torch.exp(self.log_gamma_0)

    def get_decay_rate(self) -> torch.Tensor:
        """Return decay_rate as a differentiable tensor."""
        return torch.sigmoid(self.raw_decay)

    def get_gamma_0_value(self) -> float:
        """Return gamma_0 as a Python float (for logging/inspection)."""
        return float(torch.exp(self.log_gamma_0).item())

    def get_decay_rate_value(self) -> float:
        """Return decay_rate as a Python float (for logging/inspection)."""
        return float(torch.sigmoid(self.raw_decay).item())

    def get_gamma_schedule(self) -> list:
        """
        Compute full gamma schedule as differentiable tensors.

        gamma^(k) = gamma_0 * decay_rate^k

        All gammas are connected to log_gamma_0 and raw_decay via autograd,
        so gradients flow correctly on each forward pass.

        Returns:
            gammas: list of K tensors, gamma^(k) for k=0..K-1.
                    Each is a scalar tensor with requires_grad=True.
        """
        g0 = self.get_gamma_0()
        r  = self.get_decay_rate()

        gammas = []
        for k in range(self.prop_layer_num):
            # gamma^(k) = gamma_0 * decay_rate^k
            # Using torch.pow for stability and differentiability
            gk = g0 * torch.pow(r, torch.tensor(float(k), device=g0.device))
            gammas.append(gk)

        return gammas

    def get_lam_schedule(self) -> list:
        """
        Compute full lam schedule (SCAD threshold, not zero-cutoff).

        lam^(k) = gamma^(k) / a

        This is the actual SCAD threshold parameter used in the forward pass.

        Returns:
            lams: list of K tensors, lam^(k) for k=0..K-1.
        """
        gammas = self.get_gamma_schedule()
        return [g / self.scad_a for g in gammas]

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: MLP embedding → K QN-IRLS layers with 2-parameter gamma schedule.

        Algorithm:
            1. MLP: raw features → F⁰ (class-space embeddings)
            2. For each layer k=0..K-1:
                - Compute gamma^(k) from 2-parameter schedule
                - Compute SCAD edge weights with gamma^(k)
                - QN-IRLS aggregation update

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

        # ---- 3. QN-IRLS propagation — using 2-parameter gamma schedule ----
        gammas = self.get_gamma_schedule()
        for k, gamma_k in enumerate(gammas):

            # Convert gamma to lam (zero-cutoff to SCAD threshold)
            lam_k = gamma_k / self.scad_a

            # y_{ij} = ||f_i/√d_i - f_j/√d_j||_2  (Eq. 8)
            # .detach() on Z matches RUNG convention: IRLS treats y as fixed
            # in each outer iteration (only the linear solve is differentiated).
            Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
            y = Z.sqrt()

            # W_{ij} = dρ_SCAD(y_ij)/dy²  with learnable lam_k
            # Uses torch.where — gradient flows through lam_k (not broken by masking)
            # Note: lam_k is differentiable (depends on log_gamma_0 and raw_decay)
            W = scad_weight_differentiable(y, lam_k, a=self.scad_a)

            # Zero diagonal — use out-of-place multiplication so that the
            # computation graph for gamma_k is NOT broken.
            eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            W   = W * (1.0 - eye)

            # NaN guard (isolated nodes may produce 0/0) — out-of-place
            W = torch.where(torch.isnan(W), torch.ones_like(W), W)

            # QN-IRLS update (Eq. 8 in paper):
            #   F^(k+1) = (diag(q) + λI)^{-1} [(W ⊙ Ã) F^(k) + λ F^(0)]
            # diag(q)_i = Σ_j W_ij A_ij / d_i
            Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)   # [N, 1]
            F     = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat

            # Store for inspection
            self._last_gammas[k] = gamma_k.item()

        return F

    # ------------------------------------------------------------------
    #  Feature extraction (estimation bias compatibility)
    # ------------------------------------------------------------------

    def get_aggregated_features(
        self, A: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract propagated features after all QN-IRLS layers.

        Since RUNG_parametric_gamma propagates in class space, the aggregated
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
        Yield only the schedule parameters (log_gamma_0 and raw_decay).

        Use for a separate optimizer parameter group to apply a different
        learning rate to the schedule:

            optimizer = torch.optim.Adam([
                {'params': list(model.get_non_gamma_parameters()), 'lr': 0.01},
                {'params': list(model.get_gamma_parameters()),     'lr': 0.003},
            ])

        Note: We use 0.3× for schedule LR (vs 0.2× for learnable_gamma) because
        the 2-parameter gradients are stronger and more stable.
        """
        return [self.log_gamma_0, self.raw_decay]

    def get_non_gamma_parameters(self):
        """Yield all parameters EXCEPT the schedule (gamma) parameters."""
        gamma_ids = {id(self.log_gamma_0), id(self.raw_decay)}
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
        with torch.no_grad():
            return [float(g) for g in self.get_gamma_schedule()]

    def get_learned_lams(self) -> list:
        """
        Return current learned lam (SCAD threshold) values for all layers.

        lam^(k) = exp(log_lam^(k))
        zero_cutoff^(k) = a * lam^(k)

        Returns:
            lams: list of float, length num_layers.
        """
        with torch.no_grad():
            lams = [float(l) for l in self.get_lam_schedule()]
            return lams

    def log_gamma_stats(self) -> None:
        """
        Print a formatted table of schedule parameters and resulting gammas.
        """
        print("\n" + "=" * 65)
        print(f"{'RUNG_parametric_gamma — Schedule Parameters':^65}")
        print("=" * 65)
        print(f"  γ₀ (gamma_0)          = {self.get_gamma_0_value():.6f}")
        print(f"  r  (decay_rate)       = {self.get_decay_rate_value():.6f}")
        print(f"\n  {'Layer':<8} {'gamma':>12} {'lam':>12}")
        print(f"  {'-' * 35}")
        gammas = self.get_learned_gammas()
        lams = self.get_learned_lams()
        for k in range(min(self.prop_layer_num, len(gammas))):
            print(f"  {k:<8} {gammas[k]:>12.6f} {lams[k]:>12.6f}")
        print("=" * 65 + "\n")

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
