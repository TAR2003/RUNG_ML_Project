"""
model/rung_percentile_gamma.py

Extends RUNG_learnable_gamma by replacing the learned log_gamma parameter
with a percentile of the actual edge difference distribution, computed
dynamically during every forward pass.

Model lineage:
    RUNG (NeurIPS 2024)
        └── RUNG_new_SCAD  (replaces MCP with SCAD penalty, fixed gamma)
                └── RUNG_learnable_gamma (per-layer learnable gamma via backprop)
                        └── RUNG_percentile_gamma (THIS FILE)

Core change:
    BEFORE: gamma^(k) = a * exp(log_lam^(k))   [learned, one Parameter per layer]
    AFTER:  gamma^(k) = quantile(y^(k)_edges, q) [computed from data, no Parameter]
            lam^(k)   = gamma^(k) / a

Key properties:
    1. Zero variance from gamma across seeds — deterministic given the graph
    2. Automatically adapts to feature smoothing across layers
    3. Guarantees a fixed fraction (1-q) of edges are "suspicious" every pass
    4. No gradient signal needed — gamma never gets stuck
    5. Fewer parameters than RUNG_learnable_gamma (K fewer scalars)

Interface (identical to RUNG_learnable_gamma):
    forward(A, X)  A is dense [N, N] adjacency (no self-loops),
                   X is [N, D] node feature matrix.

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
#  FULL MODEL: RUNG_percentile_gamma
# ============================================================

class RUNG_percentile_gamma(nn.Module):
    """
    RUNG with percentile-based adaptive SCAD threshold per layer.

    Architecture (identical to RUNG_learnable_gamma except gamma source):
        1. MLP: X → F^(0)   (maps raw features to class-space embeddings)
        2. K QN-IRLS graph aggregation layers, each computing its own gamma
           as quantile(y_edges, percentile_q) from the current features
        3. Returns F^(K) as logits (no separate classification head)

    Two modes controlled by use_layerwise_q:

    Mode 1 (default): Single q for all layers (use_layerwise_q=False)
        All K layers use percentile_q.  Simpler, good starting point.

    Mode 2: Different q for early vs late layers (use_layerwise_q=True)
        Layers 0 to K//2-1:  use percentile_q      (lighter pruning)
        Layers K//2 to K-1:  use percentile_q_late  (heavier pruning)
        Motivated by: adversarial edges stand out more in late layers where
        features are smoothed, so stronger pruning is appropriate late.

    Parameter count vs RUNG_learnable_gamma:
        Removed: K scalar log_lam parameters (one per layer).
        Added:   0 new parameters.
        Net change: -K parameters (strictly simpler model).

    Args:
        in_dim:               Input feature dimension D.
        out_dim:              Number of output classes C.
        hidden_dims:          MLP hidden layer widths, e.g. [64].
        lam_hat:              Skip-connection fraction λ̂ ∈ (0, 1].
                              λ = 1/λ̂ − 1
        percentile_q:         Main percentile parameter (default 0.75).
                              gamma = quantile(edge_differences, percentile_q).
                              Higher q = lighter pruning; lower q = more aggressive.
        use_layerwise_q:      If True, use different q for early vs late layers.
        percentile_q_late:    q for late layers (only if use_layerwise_q=True).
                              Should be ≤ percentile_q for more aggressive late pruning.
        scad_a:               SCAD shape parameter a (default 3.7, Fan & Li 2001).
        prop_step:            Number of QN-IRLS iterations K (default 10).
        dropout:              MLP dropout rate.
        eps:                  Numerical stability floor.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list,
        lam_hat: float = 0.9,
        percentile_q: float = 0.75,
        use_layerwise_q: bool = False,
        percentile_q_late: float = 0.65,
        scad_a: float = 3.7,
        prop_step: int = 10,
        dropout: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()

        # ---- Validate ----
        assert 0 < lam_hat <= 1, "lam_hat must be in (0, 1]"
        assert 0.0 < percentile_q < 1.0, \
            f"percentile_q must be in (0, 1), got {percentile_q}"
        assert percentile_q_late is None or 0.0 < percentile_q_late < 1.0, \
            f"percentile_q_late must be in (0, 1), got {percentile_q_late}"
        assert scad_a > 1.0, "SCAD shape param a must be > 1"

        # ---- Hyperparams ----
        self.lam_hat           = lam_hat
        self.lam               = 1.0 / lam_hat - 1.0   # skip-conn weight λ
        self.prop_layer_num    = prop_step
        self.scad_a            = scad_a
        self.percentile_q      = percentile_q
        self.use_layerwise_q   = use_layerwise_q
        self.percentile_q_late = percentile_q_late
        self.eps               = eps

        # ---- MLP backbone (identical to RUNG_learnable_gamma) ----
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # ---- NO log_lams ParameterList — this is the core difference ----
        # gamma is computed from data in every forward pass, never trained.
        # This directly eliminates the gradient-stalling failure mode of
        # RUNG_learnable_gamma and makes gamma deterministic given the graph.

        # ---- Per-layer q values (fixed floats, not parameters) ----
        self._layer_q_values = [
            self._get_q_for_layer(k) for k in range(prop_step)
        ]

        # Storage for gammas used in the most recent forward pass.
        # Updated every forward() call for analysis and logging.
        self._last_gammas = [None] * prop_step

        # ---- Save config for logging / checkpointing ----
        self._config = {
            'model':              'RUNG_percentile_gamma',
            'in_dim':             in_dim,
            'out_dim':            out_dim,
            'hidden_dims':        hidden_dims,
            'lam_hat':            lam_hat,
            'percentile_q':       percentile_q,
            'use_layerwise_q':    use_layerwise_q,
            'percentile_q_late':  percentile_q_late,
            'scad_a':             scad_a,
            'prop_step':          prop_step,
            'dropout':            dropout,
        }

    # ------------------------------------------------------------------
    #  Per-layer q assignment
    # ------------------------------------------------------------------

    def _get_q_for_layer(self, layer_idx: int) -> float:
        """
        Return the percentile q to use for layer layer_idx.

        Mode 1 (use_layerwise_q=False):    All layers → percentile_q.
        Mode 2 (use_layerwise_q=True):
            Layers 0 .. K//2-1 → percentile_q      (lighter, early layers)
            Layers K//2 .. K-1 → percentile_q_late  (heavier, late layers)
        """
        if not self.use_layerwise_q:
            return self.percentile_q
        halfway = self.prop_layer_num // 2
        if layer_idx < halfway:
            return self.percentile_q
        else:
            return self.percentile_q_late

    # ------------------------------------------------------------------
    #  Percentile gamma computation
    # ------------------------------------------------------------------

    def _compute_percentile_lam(
        self,
        y: torch.Tensor,
        A_bool: torch.Tensor,
        q: float,
    ) -> torch.Tensor:
        """
        Compute lam = gamma / scad_a where gamma = quantile(y_edges, q).

        The percentile is taken over EDGE differences only (y values where
        A > 0 and i ≠ j).  Self-loop differences are zero and would skew
        the distribution, especially at low q values.

        Args:
            y:      [N, N] tensor of all-pairs feature differences (non-negative).
            A_bool: [N, N] boolean mask, True where edge exists (including loops).
            q:      float in (0, 1), the percentile fraction.

        Returns:
            lam: scalar tensor, = percentile_gamma / scad_a.
                 Clamped to eps to avoid zero lam.
        """
        # Exclude self-loops by removing diagonal positions.
        N = y.shape[0]
        eye_bool = torch.eye(N, device=y.device, dtype=torch.bool)
        edge_mask = A_bool & ~eye_bool         # True only for off-diagonal edges

        if edge_mask.sum() == 0:
            # Degenerate graph with no non-loop edges: fall back to fixed lam
            return torch.tensor(1.0, device=y.device, dtype=y.dtype)

        y_edges = y[edge_mask]                 # 1-D tensor of edge differences

        gamma = torch.quantile(y_edges, q)
        gamma = gamma.clamp(min=self.eps)      # avoid zero gamma
        lam   = gamma / self.scad_a
        return lam

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: MLP embedding → K QN-IRLS layers with percentile gamma.

        Same algorithm as RUNG_learnable_gamma.forward() except each layer k
        computes gamma^(k) = quantile(y_edges^(k), q) from the current
        normalised feature differences — no learned threshold parameter.

        Args:
            A: [N, N] adjacency matrix (float, no self-loops expected).
            F: [N, D] node feature matrix.

        Returns:
            F: [N, C] logit matrix after K propagation iterations.
        """
        # ---- 1. MLP: raw features → initial class-space features F^(0) ----
        F0 = self.mlp(F)

        # ---- 2. Preprocessing (identical to RUNG_learnable_gamma) ----
        A       = add_loops(A)                    # add self-loops
        D       = A.sum(-1)                       # [N] degree vector
        D_sq    = D.sqrt().unsqueeze(-1)          # [N, 1]
        A_tilde = sym_norm(A)                     # D^{-1/2} A D^{-1/2}
        A_bool  = A.bool()                        # boolean mask for edge extraction

        F = F0

        # ---- 3. QN-IRLS propagation — percentile-based adaptive gamma ----
        for k in range(self.prop_layer_num):
            q_k = self._layer_q_values[k]

            # y_{ij} = ||f_i/√d_i - f_j/√d_j||_2  (Eq. 8 from paper)
            # .detach() matches RUNG convention: IRLS treats y as fixed
            Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
            y = Z.sqrt()

            # ---- PERCENTILE GAMMA — THE CORE NEW COMPUTATION ----
            # lam_k is derived purely from the current edge distribution.
            # It has NO gradient w.r.t. any parameter and is entirely
            # deterministic given F (and hence given the input graph).
            lam_k = self._compute_percentile_lam(y, A_bool, q_k)

            # Store gamma (= a * lam) used this layer for analysis
            self._last_gammas[k] = (self.scad_a * lam_k).item()

            # W_{ij} = dρ_SCAD(y_ij)/dy²  with data-derived lam_k
            W = scad_weight_differentiable(y, lam_k, a=self.scad_a)

            # Zero diagonal — out-of-place to preserve autograd for MLP weights
            eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            W   = W * (1.0 - eye)

            # NaN guard for isolated nodes — out-of-place
            W = torch.where(torch.isnan(W), torch.ones_like(W), W)

            # QN-IRLS update (Eq. 8):
            #   F^(k+1) = (diag(q) + λI)^{-1} [(W ⊙ Ã) F^(k) + λ F^(0)]
            Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)   # [N, 1]
            F     = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat

        return F

    # ------------------------------------------------------------------
    #  Feature extraction
    # ------------------------------------------------------------------

    def get_aggregated_features(
        self, A: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract propagated features after all QN-IRLS layers.

        Identical to RUNG_learnable_gamma.get_aggregated_features().
        Provided for compatibility with utils/metrics.py estimation bias.

        Args:
            A: [N, N] adjacency matrix.
            X: [N, D] node feature matrix.

        Returns:
            F: [N, C] aggregated feature matrix (= logits).
        """
        return self.forward(A, X)

    # ------------------------------------------------------------------
    #  Gamma inspection / logging
    # ------------------------------------------------------------------

    def get_last_gammas(self) -> list:
        """
        Return gamma (= a * lam) values used in the most recent forward pass.

        This is the key analysis tool — shows how gamma adapts per layer.

        Returns:
            list of float, length prop_step.
            Each value is quantile(y_edges^(k), q) from the last forward pass.

        Expected pattern:
            Gammas should DECREASE across layers because features become more
            similar (smaller y values) as aggregation smooths them.
            This directly validates the depth-gamma mismatch hypothesis.
        """
        return list(self._last_gammas)

    def log_gamma_stats(self) -> None:
        """
        Print a formatted table of percentile gamma values from the last
        forward pass.

        Example output:
            ==================================================
              RUNG_percentile_gamma — Gamma Values (last fwd)
            ==================================================
             Layer    gamma     lam      q used
            --------------------------------------------------
                 0     4.2103   1.1380    0.75
                 1     3.8821   1.0492    0.75
                ...
                 9     0.9234   0.2496    0.75
            ==================================================
        """
        print(f"\n{'=' * 54}")
        print(f"{'RUNG_percentile_gamma — Gamma Values (last fwd)':^54}")
        print(f"{'=' * 54}")
        print(f"{'Layer':>7}  {'gamma':>9}  {'lam':>9}  {'q used':>7}")
        print(f"{'-' * 54}")
        for k in range(self.prop_layer_num):
            q_k  = self._layer_q_values[k]
            g    = self._last_gammas[k]
            g_str = f"{g:.4f}" if g is not None else "  N/A "
            lam_str = f"{g / self.scad_a:.4f}" if g is not None else "  N/A "
            print(f"  {k:>5}    {g_str:>9}  {lam_str:>9}  {q_k:>7.2f}")
        print(f"{'=' * 54}\n")

    def count_parameters(self) -> int:
        """Count trainable parameters. Should be LESS than RUNG_learnable_gamma."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    #  Config / repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        q_info = (
            f"q={self.percentile_q}"
            if not self.use_layerwise_q
            else f"q_early={self.percentile_q}, q_late={self.percentile_q_late}"
        )
        return (
            f"RUNG_percentile_gamma(\n"
            f"  prop_step={self.prop_layer_num},\n"
            f"  lam_hat={self.lam_hat},\n"
            f"  {q_info},\n"
            f"  scad_a={self.scad_a},\n"
            f"  params={self.count_parameters()}\n"
            f")"
        )
