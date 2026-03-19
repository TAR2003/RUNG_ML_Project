"""
model/rung_homophily_adaptive.py

RUNG_homophily_adaptive: Automatically adjusts pruning aggressiveness
based on each node's local graph homophily.

Key insight from experiments:
    - Homophilic graphs: aggressive cosine+percentile pruning works well
    - Heterophilic graphs: aggressive pruning removes valid cross-class edges

Solution:
    - Compute soft local homophily h_i for each node from soft predictions
    - Nodes with low h_i (heterophilic neighborhood) get higher q_i
      (less aggressive pruning)
    - Nodes with high h_i (homophilic neighborhood) get lower q_i
      (more aggressive pruning)
"""

import torch
from torch import nn
import torch.nn.functional as F

from model.mlp import MLP
from model.rung_learnable_gamma import scad_weight_differentiable
from utils import add_loops, sym_norm


class RUNG_homophily_adaptive(nn.Module):
    """
    RUNG with homophily-adaptive per-node percentile gamma.

    q_i = percentile_q + (1 - h_i) * q_relax
    where h_i is node i's soft local homophily estimate.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims=None,
        lam_hat: float = 0.9,
        percentile_q: float = 0.75,
        q_relax: float = 0.20,
        q_max: float = 0.99,
        homophily_mode: str = 'from_F0',
        scad_a: float = 3.7,
        prop_step: int = 10,
        dropout: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64]

        assert 0 < lam_hat <= 1, "lam_hat must be in (0, 1]"
        assert 0.0 < percentile_q < 1.0, "percentile_q must be in (0, 1)"
        assert 0.0 <= q_relax < 1.0, "q_relax must be in [0, 1)"
        assert percentile_q <= q_max < 1.0, "q_max must be in [percentile_q, 1)"
        assert homophily_mode in ('from_F0', 'per_layer'), \
            "homophily_mode must be 'from_F0' or 'per_layer'"
        assert scad_a > 1.0, "SCAD shape param a must be > 1"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.lam_hat = lam_hat
        self.lam = 1.0 / lam_hat - 1.0
        self.percentile_q = percentile_q
        self.q_relax = q_relax
        self.q_max = q_max
        self.homophily_mode = homophily_mode
        self.scad_a = scad_a
        self.prop_layer_num = prop_step
        self.eps = eps

        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        self._last_gammas = [None] * prop_step
        self._last_h_mean = None
        self._last_q_mean = None

    def _compute_cosine_distance(self, F_norm: torch.Tensor) -> torch.Tensor:
        """Compute all-pairs cosine distance in [0, 2]."""
        F_unit = F.normalize(F_norm, p=2, dim=-1, eps=self.eps)
        cos_sim = torch.mm(F_unit, F_unit.T)
        y = (1.0 - cos_sim).clamp(min=0.0, max=2.0)
        return y

    def _compute_soft_homophily(
        self,
        F_current: torch.Tensor,
        A_bool: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_i = mean_{j in N(i)} (softmax(F_i) dot softmax(F_j)).
        Excludes diagonal self-loops.
        """
        N = F_current.shape[0]
        device = F_current.device

        P = torch.softmax(F_current, dim=-1)
        H = torch.mm(P, P.T)

        eye_bool = torch.eye(N, device=device, dtype=torch.bool)
        edge_only = A_bool & ~eye_bool

        neigh_cnt = edge_only.sum(dim=-1).float()
        sim_sum = (H * edge_only.float()).sum(dim=-1)

        min_h = 1.0 / float(self.out_dim)
        h = sim_sum / neigh_cnt.clamp(min=1.0)
        h = torch.where(neigh_cnt > 0, h, torch.full_like(h, min_h))
        h = h.clamp(min=min_h, max=1.0)
        return h

    def _compute_adaptive_q(self, h: torch.Tensor) -> torch.Tensor:
        q = self.percentile_q + (1.0 - h) * self.q_relax
        return q.clamp(min=self.percentile_q, max=self.q_max)

    def _compute_per_node_lam(
        self,
        y_full: torch.Tensor,
        A_bool: torch.Tensor,
        q_adaptive: torch.Tensor,
    ):
        """Exact per-node quantile gamma, then per-edge lam=max(lam_i, lam_j)."""
        N = y_full.shape[0]
        device = y_full.device

        gammas = torch.zeros(N, device=device, dtype=y_full.dtype)
        eye_bool = torch.eye(N, device=device, dtype=torch.bool)
        edge_only = A_bool & ~eye_bool

        all_edges = y_full[edge_only]
        global_fallback = torch.quantile(all_edges, self.percentile_q) if all_edges.numel() > 0 \
            else torch.tensor(1.0, device=device, dtype=y_full.dtype)

        for i in range(N):
            y_i = y_full[i][edge_only[i]]
            if y_i.numel() == 0:
                gammas[i] = global_fallback
            else:
                gammas[i] = torch.quantile(y_i, q_adaptive[i])

        gammas = gammas.clamp(min=self.eps)
        lam_i = gammas / self.scad_a
        lam_per_edge = torch.maximum(
            lam_i.unsqueeze(1).expand(N, N),
            lam_i.unsqueeze(0).expand(N, N),
        )
        return lam_per_edge, gammas

    def _compute_per_node_lam_fast(
        self,
        y_full: torch.Tensor,
        A_bool: torch.Tensor,
        q_adaptive: torch.Tensor,
    ):
        """
        Fast approximation for large N:
        row-wise masked sort then gather percentile index per node.
        """
        N = y_full.shape[0]
        device = y_full.device

        eye_bool = torch.eye(N, device=device, dtype=torch.bool)
        edge_only = A_bool & ~eye_bool

        degree = edge_only.sum(dim=-1)
        if degree.max().item() == 0:
            gammas = torch.ones(N, device=device, dtype=y_full.dtype)
            lam_i = gammas / self.scad_a
            lam_per_edge = torch.maximum(
                lam_i.unsqueeze(1).expand(N, N),
                lam_i.unsqueeze(0).expand(N, N),
            )
            return lam_per_edge, gammas

        # Non-edges are padded by > max cosine distance so edge distances sort first.
        y_masked = torch.where(edge_only, y_full, torch.full_like(y_full, 3.0))
        y_sorted, _ = y_masked.sort(dim=-1)

        deg_f = degree.float().clamp(min=1.0)
        idx = torch.floor(q_adaptive * (deg_f - 1.0)).long().clamp(min=0, max=N - 1)
        gammas = y_sorted.gather(1, idx.unsqueeze(1)).squeeze(1)

        # Isolated-node fallback to base gamma computed from all available edges.
        all_edges = y_full[edge_only]
        fallback = torch.quantile(all_edges, self.percentile_q) if all_edges.numel() > 0 \
            else torch.tensor(1.0, device=device, dtype=y_full.dtype)
        gammas = torch.where(degree > 0, gammas, torch.full_like(gammas, fallback))
        gammas = gammas.clamp(min=self.eps)

        lam_i = gammas / self.scad_a
        lam_per_edge = torch.maximum(
            lam_i.unsqueeze(1).expand(N, N),
            lam_i.unsqueeze(0).expand(N, N),
        )
        return lam_per_edge, gammas

    def get_last_gammas(self) -> list:
        return list(self._last_gammas)

    def get_aggregated_features(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return self.forward(A, X)

    def forward(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        F0 = self.mlp(F)

        A = add_loops(A)
        D = A.sum(-1)
        D_sq = D.sqrt().unsqueeze(-1)
        A_tilde = sym_norm(A)
        A_bool = A.bool()

        if self.homophily_mode == 'from_F0':
            with torch.no_grad():
                h = self._compute_soft_homophily(F0, A_bool)
                q_adaptive = self._compute_adaptive_q(h)
                self._last_h_mean = h.mean().item()
                self._last_q_mean = q_adaptive.mean().item()

        F = F0
        N = F.shape[0]

        for k in range(self.prop_layer_num):
            if self.homophily_mode == 'per_layer':
                with torch.no_grad():
                    h = self._compute_soft_homophily(F, A_bool)
                    q_adaptive = self._compute_adaptive_q(h)
                    self._last_h_mean = h.mean().item()
                    self._last_q_mean = q_adaptive.mean().item()

            F_norm = F / D_sq
            y = self._compute_cosine_distance(F_norm).detach()

            if N > 500:
                lam_per_edge, gammas = self._compute_per_node_lam_fast(y, A_bool, q_adaptive)
            else:
                lam_per_edge, gammas = self._compute_per_node_lam(y, A_bool, q_adaptive)
            self._last_gammas[k] = gammas.mean().item()

            W = scad_weight_differentiable(y, lam_per_edge, a=self.scad_a)

            eye = torch.eye(N, device=W.device, dtype=W.dtype)
            W = W * (1.0 - eye)
            W = torch.where(torch.isnan(W), torch.ones_like(W), W)

            Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)
            F = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat

        return F

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def log_stats(self) -> None:
        print(f"\n{'=' * 55}")
        print(f"{'RUNG_homophily_adaptive - Stats':^55}")
        print(f"{'=' * 55}")
        print(f"  percentile_q:    {self.percentile_q}")
        print(f"  q_relax:         {self.q_relax}")
        print(f"  q_max:           {self.q_max}")
        print(f"  homophily_mode:  {self.homophily_mode}")
        if self._last_h_mean is not None:
            print(f"  Last h mean:     {self._last_h_mean:.4f}")
            print(f"  Last q mean:     {self._last_q_mean:.4f}")
        if any(g is not None for g in self._last_gammas):
            gamma_str = ', '.join(f"{g:.4f}" for g in self._last_gammas if g is not None)
            print(f"  Last gamma means per layer: [{gamma_str}]")
        print(f"{'=' * 55}\n")
