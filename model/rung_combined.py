"""
model/rung_combined.py

RUNG_combined: Combines percentile gamma + cosine distance

Merges the two strongest independent improvements from this research:

    1. Percentile gamma (from RUNG_percentile_gamma)
       gamma^(k) = quantile(y^(k), q)  — auto-adapts per layer

    2. Cosine distance (from RUNG_learnable_distance mode='cosine')
       y_ij = 1 - cosine_similarity(f_i, f_j)  — scale-invariant

These target different parts of the SCAD weight computation:
    y_ij = cosine_distance(f_i, f_j)     ← cosine improvement
    gamma = quantile(y, q)                ← percentile improvement
    W_ij  = scad(y_ij, gamma)             ← both feed here

Why they stack:
    - Cosine distance is scale-invariant → y distribution is stable
      across layers (always in [0, 2] regardless of feature magnitudes)
    - Stable y distribution makes percentile gamma more meaningful:
      q=0.75 reliably means "top 25% most suspicious edges" at every layer
    - With Euclidean distance, y shrinks across layers so percentile
      is calibrated differently at each layer in an inconsistent way

New parameters vs RUNG base: ZERO (cosine is free, percentile has no params)
New parameters vs both parents: ZERO (this is purely a combination)

Model lineage:
    RUNG (NeurIPS 2024)
        ├── RUNG_percentile_gamma  (percentile gamma, Euclidean distance)
        ├── RUNG_learnable_distance (fixed gamma, cosine distance)
        └── RUNG_combined (THIS FILE — both together, no new params)
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model.mlp import MLP
from model.rung_learnable_gamma import scad_weight_differentiable
from utils import add_loops, pairwise_squared_euclidean, sym_norm


# ============================================================
#  FULL MODEL: RUNG_combined
# ============================================================

class RUNG_combined(nn.Module):
    """
    RUNG with cosine distance + percentile gamma — zero new parameters.

    Combines the strengths of:
        - RUNG_percentile_gamma: data-driven adaptive gamma
        - RUNG_learnable_distance (cosine mode): scale-invariant distance

    The full computation chain:
        1. f_norm_ij = (F_i - F_j) / sqrt(d_i)  [degree normalization]
        2. y_ij = 1 - cosine_sim(f_norm_i, f_norm_j)  [cosine distance in [0,2]]
        3. gamma = quantile(y, q)                     [percentile threshold]
        4. W_ij = scad(y_ij, gamma)                   [SCAD weights]
        5. F_new = QN-IRLS update                     [aggregation]

    Key properties:
        - Zero new parameters vs RUNG base model
        - Cosine distance: scale-invariant, range [0,2] always
        - Percentile gamma: adapts to cosine distribution at each layer
        - Together: most consistent edge suspiciousness measurement

    Two modes controlled by use_layerwise_q:

    Mode 1 (default): Single q for all layers (use_layerwise_q=False)
        All K layers use percentile_q.  Simpler, good starting point.

    Mode 2: Different q for early vs late layers (use_layerwise_q=True)
        Layers 0 to K//2-1:  use percentile_q      (lighter pruning)
        Layers K//2 to K-1:  use percentile_q_late  (heavier pruning)

    Args:
        in_dim:               Input feature dimension D.
        out_dim:              Number of output classes C.
        hidden_dims:          MLP hidden layer widths, e.g. [64].
        lam_hat:              Skip-connection fraction λ̂ ∈ (0, 1].
                              λ = 1/λ̂ − 1
        percentile_q:         Main percentile parameter (default 0.75).
                              gamma = quantile(edge_distances, percentile_q).
                              Higher q = lighter pruning; lower q = more aggressive.
                              NOTE: This is for COSINE distances [0,2], not Euclidean!
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

        # ---- MLP backbone (identical to RUNG_percentile_gamma) ----
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # ---- NO learnable distance parameters — cosine is free ----
        # Gamma is computed from data in every forward pass, never trained.

        # ---- Per-layer q values (fixed floats, not parameters) ----
        self._layer_q_values = [
            self._get_q_for_layer(k) for k in range(prop_step)
        ]

        # Storage for gammas used in the most recent forward pass.
        # Updated every forward() call for analysis and logging.
        self._last_gammas = [None] * prop_step
        self._last_y_stats = [None] * prop_step  # (mean, std, max) per layer

        # ---- Save config for logging / checkpointing ----
        self._config = {
            'model':              'RUNG_combined',
            'in_dim':             in_dim,
            'out_dim':            out_dim,
            'hidden_dims':        hidden_dims,
            'lam_hat':            lam_hat,
            'percentile_q':       percentile_q,
            'use_layerwise_q':    use_layerwise_q,
            'percentile_q_late':  percentile_q_late,
            'distance':           'cosine',
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
    #  Cosine distance computation
    # ------------------------------------------------------------------

    def _compute_cosine_distance(self, F_norm: torch.Tensor) -> torch.Tensor:
        """
        Compute all-pairs cosine distance between degree-normalized embeddings.

        y_ij = 1 - cosine_similarity(f_i, f_j)

        Range: [0, 2]
            0 = identical direction (homophilic, likely clean edge)
            1 = orthogonal (unrelated)
            2 = opposite direction (highly suspicious, likely adversarial)

        Scale-invariant: multiplying all features by a constant
        does not change any cosine distance.

        Args:
            F_norm: Degree-normalized features, [N, d]

        Returns:
            y: Cosine distances, [N, N], range [0, 2]
        """
        # L2-normalize to unit sphere (makes cosine = dot product)
        F_unit = F.normalize(F_norm, p=2, dim=-1, eps=self.eps)  # [N, d]

        # All-pairs cosine similarity: [N, N]
        cos_sim = torch.mm(F_unit, F_unit.T)  # [N, N]

        # Cosine distance = 1 - cosine_similarity
        y = 1.0 - cos_sim

        # Clamp to [0, 2] for safety (numerical noise can push slightly outside)
        y = y.clamp(min=0.0, max=2.0)

        return y

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
            y:      [N, N] tensor of cosine distances (non-negative).
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

        y_edges = y[edge_mask]                 # 1-D tensor of edge distances

        gamma = torch.quantile(y_edges, q)
        gamma = gamma.clamp(min=self.eps)      # avoid zero gamma
        lam   = gamma / self.scad_a
        return lam

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: MLP embedding → K QN-IRLS layers with cosine distance + percentile gamma.

        Combines RUNG_percentile_gamma's percentile approach with cosine distance.

        Args:
            A: [N, N] adjacency matrix (float, no self-loops expected).
            F: [N, D] node feature matrix.

        Returns:
            F: [N, C] logit matrix after K propagation iterations.
        """
        # ---- 1. MLP: raw features → initial class-space features F^(0) ----
        F0 = self.mlp(F)

        # ---- 2. Preprocessing (identical to RUNG_percentile_gamma) ----
        A       = add_loops(A)                    # add self-loops
        D       = A.sum(-1)                       # [N] degree vector
        D_sq    = D.sqrt().unsqueeze(-1)          # [N, 1]
        A_tilde = sym_norm(A)                     # D^{-1/2} A D^{-1/2}
        A_bool  = A.bool()                        # boolean mask for edge extraction

        F = F0

        # ---- 3. QN-IRLS propagation — cosine distance + percentile gamma ----
        for k in range(self.prop_layer_num):
            q_k = self._layer_q_values[k]

            # ---- COSINE DISTANCE (main combination point) ----
            F_norm = F / D_sq  # [N, d] degree-normalized
            y = self._compute_cosine_distance(F_norm)  # [N, N] in [0, 2]

            # Detach y: IRLS treats y as fixed (consistent with RUNG_percentile_gamma)
            y = y.detach()

            # Store y statistics for analysis
            y_no_diag = y[~torch.eye(y.shape[0], dtype=torch.bool, device=y.device)]
            self._last_y_stats[k] = (
                y_no_diag.mean().item(),
                y_no_diag.std().item(),
                y_no_diag.max().item(),
            )

            # ---- PERCENTILE GAMMA ----
            # gamma is set to the q-th percentile of cosine distances
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

        Provided for compatibility with utils/metrics.py.
        """
        return self.forward(A, X)

    def get_last_gammas(self) -> list:
        """Return gamma values used in the most recent forward pass."""
        return list(self._last_gammas)

    # ------------------------------------------------------------------
    #  Parameter counting and logging
    # ------------------------------------------------------------------

    def count_parameters(self):
        """Return total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def log_stats(self) -> None:
        """
        Print formatted statistics from the last forward pass.

        Shows: layer number, gamma, y distribution (mean, std, max).
        Verifies cosine distances are in [0, 2].
        """
        print(f"\n{'=' * 70}")
        print(f"{'RUNG_combined — Statistics (last fwd pass)':^70}")
        print(f"{'=' * 70}")
        print(f"  distance:      cosine (scale-invariant)")
        print(f"  percentile_q:  {self.percentile_q}")
        print(f"  parameters:    {self.count_parameters()}")
        print(f"\n  {'Layer':>6}  {'gamma':>9}  "
              f"{'y_mean':>9}  {'y_std':>9}  {'y_max':>9}")
        print(f"  {'─' * 60}")
        for k, (g, ys) in enumerate(zip(self._last_gammas, self._last_y_stats)):
            if g is not None and ys is not None:
                print(f"  {k:>6}  {g:>9.4f}  "
                      f"{ys[0]:>9.4f}  {ys[1]:>9.4f}  {ys[2]:>9.4f}")
        print(f"{'=' * 70}\n")

    # ------------------------------------------------------------------
    #  String representations
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"RUNG_combined(\n"
            f"  distance=cosine,\n"
            f"  use_layerwise_q={self.use_layerwise_q},\n"
            f"  percentile_q={self.percentile_q},\n"
            f"  num_layers={self.prop_layer_num},\n"
            f"  lam_hat={self.lam_hat},\n"
            f"  parameters={self.count_parameters()}\n"
            f")"
        )

    def extra_repr(self):
        return (f'distance=cosine, '
                f'percentile_q={self.percentile_q}, '
                f'num_layers={self.prop_layer_num}, '
                f'params={self.count_parameters()}')
