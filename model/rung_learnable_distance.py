"""
model/rung_learnable_distance.py

Extends RUNG_percentile_gamma by replacing the fixed Euclidean distance
with a configurable distance metric for computing edge suspiciousness scores.

Three distance modes:
    'cosine':     1 - cosine_similarity  (0 new parameters) — START HERE
    'projection': distance in MLP-projected space  (small MLP, ~hidden*proj_dim params)
    'bilinear':   Mahalanobis-style distance after learned linear projection

Model lineage:
    RUNG (NeurIPS 2024)
        └── RUNG_learnable_gamma (per-layer learnable gamma)
                └── RUNG_percentile_gamma (percentile-based gamma)
                        └── RUNG_learnable_distance (THIS FILE — configurable distance)

Core change from RUNG_percentile_gamma:
    BEFORE: y_ij = ||f_i/√d_i - f_j/√d_j||_2         [Euclidean, fixed]
    AFTER:  y_ij = distance_module(f_i/√d_i, f_j/√d_j)  [configurable]

Why better distance helps:
    RUNG's entire defense depends on y_ij being LARGE for adversarial edges
    (cross-class connections). With Euclidean distance, an attacker can find
    "invisible" edges: features with small Euclidean distance but large cosine
    distance (different directions, similar norms). Cosine distance is
    direction-based, not magnitude-based, making such invisible edges impossible.

Key insight on gamma re-tuning:
    The percentile-based gamma automatically adapts to the distance scale:
        Euclidean range: 0 to 5+ (depends on embedding norms)
        Cosine range:    0 to 2  (scale-invariant by definition)
    Percentile-based gamma handles this automatically, so no manual re-tuning
    is needed when switching distance modes.

Do NOT modify RUNG, RUNG_new_SCAD, RUNG_learnable_gamma, or
RUNG_percentile_gamma in any way.

Reference: "Robust Graph Neural Networks via Unbiased Aggregation" NeurIPS 2024
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model.mlp import MLP
from model.rung_learnable_gamma import scad_weight_differentiable
from utils import add_loops, pairwise_squared_euclidean, sym_norm


# ============================================================
#  DISTANCE MODULE
# ============================================================

class DistanceModule(nn.Module):
    """
    Computes pairwise distance between ALL pairs of node embeddings.

    Three modes:
        'cosine':     1 - cosine_similarity ∈ [0, 2], no parameters
        'projection': L2 distance in learned lower-dimensional space
        'bilinear':   L2 distance after learned linear transformation

    Important: This is NOT a learnable similarity metric that learns an
    embedding space. It operates on FIXED degree-normalized features from
    the current layer and computes a distance matrix. Mode B/C add learned
    transformations that are applied BEFORE distance computation.

    Args:
        hidden_dim:  Dimension of input embeddings (e.g., 64)
        mode:        Distance function ('cosine', 'projection', or 'bilinear')
        proj_dim:    Output dimension for projection/bilinear modes.
                     Ignored for cosine mode.
    """

    def __init__(self, hidden_dim, mode='cosine', proj_dim=32):
        super().__init__()

        assert mode in ('cosine', 'projection', 'bilinear'), \
            f"mode must be 'cosine', 'projection', or 'bilinear', got {mode}"

        self.mode       = mode
        self.proj_dim   = proj_dim
        self.hidden_dim = hidden_dim

        if mode == 'projection':
            # Small 2-layer MLP: hidden_dim → hidden_dim//2 → proj_dim
            mid = max(hidden_dim // 2, proj_dim)
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, mid),
                nn.ReLU(),
                nn.Linear(mid, proj_dim),
            )

        elif mode == 'bilinear':
            # Linear projection: hidden_dim → proj_dim (no bias for Mahalanobis)
            self.W = nn.Linear(hidden_dim, proj_dim, bias=False)

        # cosine mode: no parameters

    def forward(self, F_norm):
        """
        Compute all-pairs pairwise distance.

        Args:
            F_norm: Degree-normalized features, shape [N, hidden_dim]
                    (precomputed: F_norm = F / sqrt(D))

        Returns:
            y: All-pairs distance matrix, shape [N, N], non-negative.
               Larger values = more suspicious = more likely to be pruned.
               y[i, j] = distance(f_i, f_j)
        """
        if self.mode == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            # cosine_similarity = (f_i · f_j) / (||f_i|| * ||f_j||)
            # Normalize each node embedding to unit sphere
            F_unit = F.normalize(F_norm, p=2, dim=-1, eps=1e-8)  # [N, d]

            # All-pairs cosine similarity: [N, N]
            cos_sim = torch.mm(F_unit, F_unit.T)  # [N, N]

            # Cosine distance = 1 - cosine_similarity
            y = 1.0 - cos_sim

            # Clamp to [0, 2] for safety (numerical noise can push slightly outside)
            y = y.clamp(min=0.0, max=2.0)

        elif self.mode == 'projection':
            # Project to smaller space, normalize, compute L2 distance
            H = self.proj(F_norm)  # [N, proj_dim]
            H_unit = F.normalize(H, p=2, dim=-1, eps=1e-8)

            # All-pairs L2 distance in projected space
            # ||h_i - h_j||_2 = sqrt(||h_i||^2 - 2*h_i·h_j + ||h_j||^2)
            # Since h_i are unit vectors: = sqrt(2 - 2*h_i·h_j) = sqrt(2(1 - h_i·h_j))
            H_sq_norm = (H_unit ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
            H_dot = torch.mm(H_unit, H_unit.T)  # [N, N]
            y_sq = H_sq_norm + H_sq_norm.T - 2.0 * H_dot  # [N, N]
            y = y_sq.sqrt().clamp(min=0.0)  # Clamp for numerical safety

        elif self.mode == 'bilinear':
            # Linear transform then L2 distance
            H = self.W(F_norm)  # [N, proj_dim]

            # All-pairs L2 distance: ||h_i - h_j||_2
            H_sq_norm = (H ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
            H_dot = torch.mm(H, H.T)  # [N, N]
            y_sq = H_sq_norm + H_sq_norm.T - 2.0 * H_dot  # [N, N]
            y = y_sq.sqrt().clamp(min=0.0)

        return y

    def count_parameters(self):
        """Return number of learnable parameters in this module."""
        return sum(p.numel() for p in self.parameters())


# ============================================================
#  FULL MODEL: RUNG_learnable_distance
# ============================================================

class RUNG_learnable_distance(nn.Module):
    """
    RUNG with configurable distance metric for edge suspiciousness.

    Builds on RUNG_percentile_gamma by replacing the hardcoded Euclidean
    distance with a pluggable DistanceModule that supports cosine, projection,
    or bilinear distance.

    CRITICAL: When you change distance mode, the range of y_ij changes:
        Euclidean:   range 0 to 5+ (depends on embedding norms)
        Cosine:      range 0 to 2  (scale-free by definition)
        Projection:  range 0 to 2  (unit vectors)
        Bilinear:    range 0+ (depends on W norm)

    However, percentile-based gamma AUTOMATICALLY adapts to any range by
    taking the q-th percentile of y values. So no manual re-tuning is needed
    when switching distance modes — same q and training procedure works.

    Recommended experiment order:
        1. RUNG_learnable_distance with distance_mode='cosine'   (baseline, 0 params)
        2. RUNG_learnable_distance with distance_mode='projection'  (if cosine improves)
        3. Compare all three against RUNG_percentile_gamma (Euclidean baseline)

    Args:
        in_dim:          Input feature dimension D
        out_dim:         Number of output classes C
        hidden_dims:     MLP hidden layer widths, e.g., [64]
        lam_hat:         Skip-connection fraction λ̂ ∈ (0, 1]
        percentile_q:    Main percentile parameter (default 0.75)
        use_layerwise_q: If True, use different q for early vs late layers
        percentile_q_late: Percentile q for late layers (if use_layerwise_q=True)
        distance_mode:   'cosine', 'projection', or 'bilinear'
        proj_dim:        Projection dimension for projection/bilinear modes
        scad_a:          SCAD shape parameter (default 3.7)
        prop_step:       Number of QN-IRLS iterations K (default 10)
        dropout:         MLP dropout rate
        eps:             Numerical stability floor
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
        distance_mode: str = 'cosine',
        proj_dim: int = 32,
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
        assert distance_mode in ('cosine', 'projection', 'bilinear'), \
            f"distance_mode must be cosine/projection/bilinear, got {distance_mode}"

        # ---- Hyperparams ----
        self.lam_hat           = lam_hat
        self.lam               = 1.0 / lam_hat - 1.0   # skip-conn weight λ
        self.prop_layer_num    = prop_step
        self.scad_a            = scad_a
        self.percentile_q      = percentile_q
        self.use_layerwise_q   = use_layerwise_q
        self.percentile_q_late = percentile_q_late
        self.distance_mode     = distance_mode
        self.proj_dim          = proj_dim
        self.eps               = eps

        # ---- MLP backbone ----
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # ---- DISTANCE MODULE (the new component) ----
        # Feature dimension after MLP is out_dim (same as number of classes)
        self.distance = DistanceModule(
            hidden_dim=out_dim,
            mode=distance_mode,
            proj_dim=proj_dim,
        )

        # ---- Per-layer q values ----
        self._layer_q_values = [
            self._get_q_for_layer(k) for k in range(prop_step)
        ]

        # Storage for analysis
        self._last_gammas   = [None] * prop_step
        self._last_y_stats  = [None] * prop_step  # (mean, std, max) per layer

        # ---- Save config for logging ----
        self._config = {
            'model':              'RUNG_learnable_distance',
            'in_dim':             in_dim,
            'out_dim':            out_dim,
            'hidden_dims':        hidden_dims,
            'lam_hat':            lam_hat,
            'percentile_q':       percentile_q,
            'use_layerwise_q':    use_layerwise_q,
            'percentile_q_late':  percentile_q_late,
            'distance_mode':      distance_mode,
            'proj_dim':           proj_dim,
            'scad_a':             scad_a,
            'prop_step':          prop_step,
            'dropout':            dropout,
        }

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

    def _compute_percentile_lam(
        self,
        y: torch.Tensor,
        A_bool: torch.Tensor,
        q: float,
    ) -> torch.Tensor:
        """
        Compute lam = gamma / scad_a where gamma = quantile(y_edges, q).

        Takes percentile over edges only (ignoring diagonals).

        Args:
            y:      [N, N] tensor of all-pairs distances (non-negative).
            A_bool: [N, N] boolean mask, True where edge exists.
            q:      float in (0, 1), the percentile fraction.

        Returns:
            lam: scalar tensor, = percentile_gamma / scad_a.
        """
        # Exclude self-loops
        N = y.shape[0]
        eye_bool = torch.eye(N, device=y.device, dtype=torch.bool)
        edge_mask = A_bool & ~eye_bool

        if edge_mask.sum() == 0:
            return torch.tensor(1.0, device=y.device, dtype=y.dtype)

        y_edges = y[edge_mask]
        gamma = torch.quantile(y_edges, q)
        gamma = gamma.clamp(min=self.eps)
        lam = gamma / self.scad_a
        return lam

    def forward(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: MLP embedding → K QN-IRLS layers with configurable distance.

        Same algorithm as RUNG_percentile_gamma except the distance computation
        is delegated to self.distance() instead of hardcoded Euclidean distance.

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

        # ---- 3. QN-IRLS propagation with configurable distance ----
        for k in range(self.prop_layer_num):
            q_k = self._layer_q_values[k]

            # ---- CONFIGURABLE DISTANCE COMPUTATION (the new part) ----
            # Degree-normalize features
            F_norm = F / D_sq  # [N, d]

            # Compute all-pairs distance using the configured distance module
            # IMPORTANT: Do NOT detach if distance module has learnable parameters
            # (projection/bilinear modes). For cosine mode (0 params), detach is harmless.
            y = self.distance(F_norm)  # [N, N]
            
            # For cosine mode (0 parameters): detach doesn't matter
            # For projection/bilinear modes (learnable): don't detach to enable gradient flow
            # We conditionally detach only if distanceModule has parameters to optimize
            if self.distance.count_parameters() == 0:
                y = y.detach()  # Cosine mode: no learnable params, safe to detach

            # Store y statistics for analysis
            y_no_diag = y[~torch.eye(y.shape[0], dtype=torch.bool, device=y.device)]
            self._last_y_stats[k] = (
                y_no_diag.mean().item(),
                y_no_diag.std().item(),
                y_no_diag.max().item(),
            )

            # ---- PERCENTILE GAMMA ----
            lam_k = self._compute_percentile_lam(y, A_bool, q_k)
            self._last_gammas[k] = (self.scad_a * lam_k).item()

            # W_{ij} = dρ_SCAD(y_ij)/dy²  with percentile-based lam_k
            W = scad_weight_differentiable(y, lam_k, a=self.scad_a)

            # Zero diagonal
            eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            W = W * (1.0 - eye)

            # NaN guard
            W = torch.where(torch.isnan(W), torch.ones_like(W), W)

            # QN-IRLS update
            Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)
            F = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat

        return F

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

    def get_distance_parameters(self):
        """Return only distance module parameters for separate learning rate."""
        return list(self.distance.parameters())

    def get_non_distance_parameters(self):
        """Return all parameters except distance module."""
        dist_ids = {id(p) for p in self.distance.parameters()}
        return [p for p in self.parameters() if id(p) not in dist_ids]

    def log_stats(self) -> None:
        """
        Print formatted statistics from the last forward pass.

        Shows: layer number, gamma, y distribution (mean, std, max).
        """
        print(f"\n{'=' * 70}")
        print(f"{'RUNG_learnable_distance — Statistics (last fwd pass)':^70}")
        print(f"{'=' * 70}")
        print(f"  distance_mode: {self.distance_mode}")
        print(f"  percentile_q:  {self.percentile_q}")
        print(f"  distance_params: {self.distance.count_parameters()}")
        print(f"\n  {'Layer':>6}  {'gamma':>9}  "
              f"{'y_mean':>9}  {'y_std':>9}  {'y_max':>9}")
        print(f"  {'─' * 60}")
        for k, (g, ys) in enumerate(zip(self._last_gammas, self._last_y_stats)):
            if g is not None and ys is not None:
                print(f"  {k:>6}  {g:>9.4f}  "
                      f"{ys[0]:>9.4f}  {ys[1]:>9.4f}  {ys[2]:>9.4f}")
        print(f"{'=' * 70}\n")

    def count_parameters(self):
        """Return total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
