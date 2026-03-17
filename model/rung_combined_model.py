"""
model/rung_combined_model.py

RUNG_combined_model: Integrates three complementary improvements:

    1. Cosine Distance (from RUNG_learnable_distance)
       y_ij = 1 - cosine_similarity(f_tilde_i, f_tilde_j)
       Scale-invariant, always in [0,2], catches cross-class edges

    2. Parametric Gamma (from RUNG_parametric_gamma)
       gamma_param^(k) = gamma_0 * decay_rate^k
       Smooth geometric decay learned via 2 shared parameters

    3. Percentile Gamma (from RUNG_percentile_gamma)
       gamma_data^(k) = quantile(y_edges, q)
       Data-driven adaptation, no parameters, deterministic

    Combined via learnable blending:
       gamma^(k) = alpha * gamma_param^(k) + (1-alpha) * gamma_data^(k)
       alpha = sigmoid(raw_alpha_blend)    ← 1 new learnable scalar

Total new parameters vs RUNG base: 3 scalars
    log_gamma_0, raw_decay, raw_alpha_blend

Model lineage:
    RUNG (NeurIPS 2024)
        ├── RUNG_percentile_gamma   (percentile gamma)
        ├── RUNG_parametric_gamma   (2-param schedule)
        ├── RUNG_learnable_distance (cosine distance)
        └── RUNG_combined_model     (THIS FILE — all three together)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mlp import MLP
from model.rung_learnable_gamma import scad_weight_differentiable
from utils import add_loops, pairwise_squared_euclidean, sym_norm


# ============================================================
#  COMBINED LAYER
# ============================================================

class RUNGLayer_combined_model(nn.Module):
    """
    Single RUNG aggregation layer combining all three mechanisms.

    Cosine distance + Parametric gamma + Percentile gamma + learnable blend.

    This layer has NO new parameters of its own.
    All learnable parameters (log_gamma_0, raw_decay, raw_alpha_blend)
    live in the parent RUNG_combined_model and are passed in at each
    forward call. This matches the pattern used by RUNG_parametric_gamma.

    Args:
        percentile_q:   Float in (0,1). Percentile for data-driven gamma.
                        Default 0.75 (same as RUNG_percentile_gamma default).
        scad_a:         SCAD shape parameter. Default 3.7.
        layer_idx:      Layer index for logging.
        eps:            Numerical stability floor.
    """

    def __init__(
        self,
        percentile_q=0.75,
        scad_a=3.7,
        layer_idx=0,
        eps=1e-8,
    ):
        super().__init__()
        self.percentile_q = percentile_q
        self.scad_a       = scad_a
        self.layer_idx    = layer_idx
        self.eps          = eps

    def forward(
        self,
        F_current,
        F0,
        A,
        D,
        D_sq,
        A_tilde,
        A_bool,
        lam_hat,
        gamma_param_k,
        alpha_blend,
    ):
        """
        One QN-IRLS step with combined gamma and cosine distance.

        Args:
            F_current:    Current node features at layer k, shape [N, d]
            F0:           Initial features (skip connection), shape [N, d]
            A:            Adjacency matrix with self-loops, shape [N, N]
            D:            Degree vector [N]
            D_sq:         sqrt(D) as [N, 1]
            A_tilde:      Symmetric normalized adjacency D^{-1/2}AD^{-1/2}
            A_bool:       Boolean adjacency mask
            lam_hat:      Skip connection fraction
            gamma_param_k: Parametric gamma for this layer (differentiable tensor)
                           = gamma_0 * decay_rate^k, computed in parent model
            alpha_blend:  Blending coefficient (differentiable tensor)
                          = sigmoid(raw_alpha_blend), computed in parent model

        Returns:
            F_new:        Updated features, shape [N, d]
        """
        # ---- Degree normalization ----
        # f_tilde = f / sqrt(d)  for use in distance computation
        F_norm = F_current / D_sq  # [N, d]

        # ---- Cosine distance computation ----
        # y_ij = 1 - cosine_similarity(f_i, f_j)  in [0, 2]
        # (cosine similarity is scale-invariant)
        F_unit = F.normalize(F_norm, p=2, dim=-1, eps=self.eps)  # [N, d]
        cos_sim = torch.mm(F_unit, F_unit.T)  # [N, N]
        y = (1.0 - cos_sim).clamp(min=0.0, max=2.0)  # Clamp for numerical safety

        # Detach y since we don't optimize through the cosine distance
        # (it's deterministic given F)
        y = y.detach()

        # ---- Percentile gamma (data-driven) ----
        N = y.shape[0]
        eye_bool = torch.eye(N, dtype=torch.bool, device=y.device)
        edge_mask = A_bool & ~eye_bool
        
        if edge_mask.sum() > 0:
            y_edges = y[edge_mask]
            gamma_data_k = torch.quantile(y_edges, self.percentile_q)
            gamma_data_k = gamma_data_k.clamp(min=self.eps)
        else:
            gamma_data_k = torch.tensor(1.0, device=y.device, dtype=y.dtype)

        lam_data_k = gamma_data_k / self.scad_a

        # ---- Parametric gamma received from parent ----
        lam_param_k = gamma_param_k / self.scad_a

        # ---- Blend the two gammas via learnable alpha ----
        lam_combined_k = alpha_blend * lam_param_k + (1.0 - alpha_blend) * lam_data_k

        # ---- SCAD edge weights ----
        W = scad_weight_differentiable(y, lam_combined_k, a=self.scad_a)

        # Zero diagonal  (convert eye_bool to float for multiplication)
        W = W * (1.0 - eye_bool.float())

        # NaN guard
        W = torch.where(torch.isnan(W), torch.ones_like(W), W)

        # ---- QN-IRLS aggregation update ----
        # F^(k+1) = (diag(q) + λI)^{-1} [(W ⊙ Ã) F^(k) + λ F^(0)]
        lam = 1.0 / lam_hat - 1.0
        Q_hat = ((W * A).sum(-1) / D + lam).unsqueeze(-1)  # [N, 1]
        F_new = (W * A_tilde) @ F_current / Q_hat + lam * F0 / Q_hat

        # Store gammas for logging
        gamma_combined = lam_combined_k * self.scad_a
        return F_new, gamma_combined.item()


# ============================================================
#  FULL MODEL
# ============================================================

class RUNG_combined_model(nn.Module):
    """
    RUNG with Cosine Distance + Parametric Gamma + Percentile Gamma.

    Architecture identical to RUNG base.
    Training: same as parametric_gamma (two-group optimizer for schedule params).

    New parameters vs RUNG base:
        log_gamma_0:      Initial gamma magnitude (1 scalar)
        raw_decay:        Decay rate per layer (1 scalar)
        raw_alpha_blend:  Parametric/percentile blend weight (1 scalar)
        Total: 3 scalar parameters

    Args:
        in_dim:           Input feature dimension
        out_dim:          Number of output classes
        hidden_dims:      MLP hidden layer widths (e.g. [64])
        lam_hat:          Skip connection fraction in (0,1)
        percentile_q:     Percentile for data-driven gamma (default 0.75)
        gamma_0_init:     Initial gamma_0 value (default 3.0)
        decay_rate_init:  Initial decay rate (default 0.85)
        alpha_blend_init: Initial blend weight, 0=percentile, 1=parametric
                          (default 0.5 = equal blend to start)
        scad_a:           SCAD shape parameter (default 3.7)
        prop_step:        Number of QN-IRLS layers K (default 10)
        dropout:          MLP dropout rate (default 0.5)
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims=None,
        lam_hat=0.9,
        percentile_q=0.75,
        gamma_0_init=3.0,
        decay_rate_init=0.85,
        alpha_blend_init=0.5,
        scad_a=3.7,
        prop_step=10,
        dropout=0.5,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64]

        self.in_dim          = in_dim
        self.out_dim         = out_dim
        self.hidden_dims     = hidden_dims
        self.lam_hat         = lam_hat
        self.percentile_q    = percentile_q
        self.scad_a          = scad_a
        self.prop_step       = prop_step
        self.dropout         = dropout

        # ---- THE THREE LEARNABLE PARAMETERS ----

        # From parametric_gamma: geometric decay schedule
        self.log_gamma_0 = nn.Parameter(
            torch.tensor(float(np.log(gamma_0_init)))
        )
        logit_decay = float(np.log(decay_rate_init / (1.0 - decay_rate_init)))
        self.raw_decay = nn.Parameter(
            torch.tensor(logit_decay)
        )

        # New: blend weight between parametric and percentile
        logit_alpha = float(np.log(alpha_blend_init / (1.0 - alpha_blend_init + 1e-8) + 1e-8))
        self.raw_alpha_blend = nn.Parameter(
            torch.tensor(logit_alpha)
        )

        # ---- MLP ENCODER ----
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # ---- PROPAGATION LAYERS ----
        self.prop_layers = nn.ModuleList([
            RUNGLayer_combined_model(
                percentile_q=percentile_q,
                scad_a=scad_a,
                layer_idx=k,
            )
            for k in range(prop_step)
        ])

        # Analysis storage
        self._last_gammas      = [None] * prop_step
        self._last_alpha_blend = None

        # ---- Save config for logging / checkpointing ----
        self._config = {
            'model':              'RUNG_combined_model',
            'in_dim':             in_dim,
            'out_dim':            out_dim,
            'hidden_dims':        hidden_dims,
            'lam_hat':            lam_hat,
            'percentile_q':       percentile_q,
            'gamma_0_init':       gamma_0_init,
            'decay_rate_init':    decay_rate_init,
            'alpha_blend_init':   alpha_blend_init,
            'scad_a':             scad_a,
            'prop_step':          prop_step,
            'dropout':            dropout,
        }

    # ---- SCHEDULE PROPERTIES ----

    @property
    def gamma_0(self):
        """Current gamma_0 value (read-only, for logging)."""
        return torch.exp(self.log_gamma_0).item()

    @property
    def decay_rate(self):
        """Current decay rate (read-only, for logging)."""
        return torch.sigmoid(self.raw_decay).item()

    @property
    def alpha_blend(self):
        """Current blend weight (read-only, for logging)."""
        return torch.sigmoid(self.raw_alpha_blend).item()

    def get_gamma_schedule(self):
        """
        Compute full parametric gamma schedule as differentiable tensors.

        Returns list of K tensors: [gamma^(0), gamma^(1), ..., gamma^(K-1)]
        All connected to log_gamma_0 and raw_decay via autograd.
        """
        g0 = torch.exp(self.log_gamma_0)
        r  = torch.sigmoid(self.raw_decay)

        gammas = []
        for k in range(self.prop_step):
            gk = g0 * torch.pow(r, torch.tensor(float(k), device=g0.device))
            gammas.append(gk)

        return gammas

    def get_gamma_parameters(self):
        """Return schedule + blend parameters for separate LR group."""
        return [self.log_gamma_0, self.raw_decay, self.raw_alpha_blend]

    def get_non_gamma_parameters(self):
        """Return all parameters EXCEPT schedule + blend parameters."""
        schedule_ids = {id(self.log_gamma_0), id(self.raw_decay),
                        id(self.raw_alpha_blend)}
        return [p for p in self.parameters() if id(p) not in schedule_ids]

    def forward(self, A, X):
        """
        Full forward pass with all three mechanisms.

        Computes gamma_schedule once, then applies per-layer with
        cosine distance and percentile gamma blend at each layer.

        Args:
            A:  Adjacency matrix [N, N]
            X:  Node features, shape [N, in_dim]

        Returns:
            logits: shape [N, out_dim]
        """
        # ---- Compute schedule (differentiable) ----
        gamma_schedule = self.get_gamma_schedule()     # list of K tensors
        alpha          = torch.sigmoid(self.raw_alpha_blend)  # scalar tensor
        self._last_alpha_blend = alpha.item()

        # ---- Preprocessing ----
        A = add_loops(A)
        D = A.sum(-1)  # [N]
        D_sq = D.sqrt().unsqueeze(-1)  # [N, 1]
        A_tilde = sym_norm(A)  # D^{-1/2} A D^{-1/2}
        A_bool = A.bool()

        # ---- MLP encoding ----
        F0 = self.mlp(X)  # [N, out_dim]

        # ---- K aggregation layers ----
        F = F0.clone()
        for k, layer in enumerate(self.prop_layers):
            F, gamma_used = layer(
                F_current    = F,
                F0           = F0,
                A            = A,
                D            = D,
                D_sq         = D_sq,
                A_tilde      = A_tilde,
                A_bool       = A_bool,
                lam_hat      = self.lam_hat,
                gamma_param_k = gamma_schedule[k],
                alpha_blend  = alpha,
            )
            self._last_gammas[k] = gamma_used

        return F

    def get_aggregated_features(self, A, X):
        """
        Extract propagated features after all QN-IRLS layers.

        Provided for compatibility with estimation bias computation.
        """
        return self.forward(A, X)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def log_stats(self):
        """Print schedule, blend, and per-layer gammas."""
        print(f"\n{'='*55}")
        print(f"{'RUNG_combined_model — Stats':^55}")
        print(f"{'='*55}")
        print(f"  gamma_0:     {self.gamma_0:.4f}")
        print(f"  decay_rate:  {self.decay_rate:.4f}")
        print(f"  alpha_blend: {self.alpha_blend:.4f}  "
              f"({'parametric' if self.alpha_blend > 0.5 else 'percentile'} dominant)")
        print(f"  percentile_q: {self.percentile_q}")
        print(f"  parameters:  {self.count_parameters()}")
        print(f"\n  {'Layer':>5}  {'gamma':>8}")
        print(f"  {'─'*20}")
        for k, g in enumerate(self._last_gammas):
            if g is not None:
                print(f"  {k:>5}  {g:>8.4f}")
        print(f"{'='*55}\n")

    def __repr__(self):
        return (
            f"RUNG_combined_model(\n"
            f"  gamma_0={self.gamma_0:.3f}, decay_rate={self.decay_rate:.3f},\n"
            f"  alpha_blend={self.alpha_blend:.3f} "
            f"({'parametric' if self.alpha_blend > 0.5 else 'percentile'} dominant),\n"
            f"  percentile_q={self.percentile_q}, distance=cosine,\n"
            f"  prop_step={self.prop_step}, params={self.count_parameters()}\n"
            f")"
        )
