import torch
from torch import nn

from model.mlp import MLP
from model.penalty import PenaltyFunction
from utils import add_loops, pairwise_squared_euclidean, sym_norm


class RUNG(nn.Module):
    """
    Robust Unbiased Graph Neural Network (RUNG) — NeurIPS 2024.

    Decoupled architecture:
        1. MLP: X  →  F⁰  (initial features in class space)
        2. QN-IRLS propagation: K iterations of robust graph smoothing
        3. Returns F^(K) as logits (no separate classification head)

    The propagation minimises the RUGE objective (Eq. 3 in paper):
        H(F) = Σ_{(i,j)∈E} ρ_γ(||f_i/√d_i - f_j/√d_j||) + λ Σ_i ||f_i - f_i⁰||²

    Penalty choices (via `penalty` argument):
        'mcp'      — Minimax Concave Penalty (default, paper's RUNG)
        'scad'     — Smoothly Clipped Absolute Deviation
        'l1'       — L1 / geometric-median (biased baseline)
        'l2'       — L2 (reduces to APPNP / GCN)
        'adaptive' — Homophily-aware weight for heterophilic graphs

    Args:
        in_dim:       Input feature dimension
        out_dim:      Number of classes
        hidden_dims:  MLP hidden layer widths, e.g. [64]
        w_func:       Edge weight callable w(y) -> W  [N, N].
                      Used when penalty is None or 'mcp'/'scad'/'l1'/'l2'
                      (the caller is responsible for providing the right function).
                      Ignored when penalty='adaptive'.
        lam_hat:      Skip-connection strength in (0, 1].
                      λ = 1/lam_hat - 1  (larger lam_hat → stronger smoothing)
        quasi_newton: Use QN-IRLS (True, default) or gradient IRLS (False)
        eta:          Step-size for gradient IRLS (only used when quasi_newton=False)
        prop_step:    Number of propagation iterations K
        dropout:      MLP dropout rate
        penalty:      Penalty name ('mcp'|'scad'|'l1'|'l2'|'adaptive'|None).
                      When None, `w_func` is used directly (backward-compatible).
        gamma:        Penalty threshold γ (MCP) / λ (SCAD).
                      Only used when penalty='adaptive' (other modes read γ from w_func).
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_dims: list,
            w_func: callable,
            lam_hat: float,
            quasi_newton: bool = True,
            eta=None,
            prop_step: int = 10,
            dropout: float = 0.5,
            penalty: str = None,
            gamma: float = 3.0,
    ):
        super().__init__()

        # MLP backbone
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # Graph smoothing objective:
        #   Σ_edge ρ(||fi - fj||) + λ Σ_node ||fi - fi0||²
        self.lam_hat = lam_hat
        self.lam = 1.0 / lam_hat - 1.0   # λ = 1/lam_hat - 1
        self.quasi_newton = quasi_newton
        self.prop_layer_num = prop_step
        self.w: callable = w_func         # W_ij = dρ(y)/dy²
        self.eta = eta

        # Extended penalty configuration
        self.penalty = penalty            # None → use w_func directly
        self.gamma = gamma                # threshold for adaptive mode

        # Validate
        assert 0 < lam_hat <= 1, 'lam_hat must be in (0, 1]'
        if quasi_newton:
            assert eta is None, 'QN-IRLS does not use a manual step-size (eta)'
        else:
            assert eta is not None and eta > 0, 'Gradient IRLS requires eta > 0'

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, A, F):
        """
        Forward pass: MLP embedding followed by K QN-IRLS propagation steps.

        Args:
            A: [N, N] adjacency matrix (float, no self-loops expected)
            F: [N, D] node feature matrix

        Returns:
            F: [N, C] logit matrix after K propagation iterations
        """
        # 1. MLP: raw features → initial class-space features F⁰
        F0 = self.mlp(F)

        # 2. Preprocessing
        A = add_loops(A)                        # add self-loop to avoid zero degree
        D = A.sum(-1)                           # [N] degree vector
        D_sq = D.sqrt().unsqueeze(-1)           # [N, 1]
        A_tilde = sym_norm(A)                   # D^{-1/2} A D^{-1/2}

        F = F0

        # 3. QN-IRLS propagation
        for _layer in range(self.prop_layer_num):
            # y_{ij} = ||f_i/√d_i - f_j/√d_j||_2  (pairwise feature diffs)
            Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
            y = Z.sqrt()

            # W_{ij} = dρ(y)/dy²  (edge reweighting)
            if self.penalty == 'adaptive':
                # Homophily-aware weight: use current soft class predictions
                soft_labels = torch.softmax(F.detach(), dim=-1)  # [N, C]
                W = PenaltyFunction.homophily_adaptive(y, self.gamma, soft_labels)
            else:
                # Standard penalties: use the callable w_func
                W = self.w(y)

            # Zero diagonal (Remark 2 in paper: self-edges excluded from Σ_edge)
            idx = torch.arange(W.shape[0], device=W.device)
            W[idx, idx] = 0.0
            W[torch.isnan(W)] = 1.0   # NaN guard (e.g. 0/0 for isolated nodes)

            if self.quasi_newton:
                # QN-IRLS update (Eq. 8 in paper) — convergence guaranteed
                Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)  # [N, 1]
                F = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat
            else:
                # Gradient IRLS update
                diag_q = torch.diag((W * A).sum(-1)) / D
                grad_smoothing = 2.0 * (diag_q - W * A_tilde) @ F
                grad_reg = 2.0 * self.lam * (F - F0)
                F = F - self.eta * (grad_smoothing + grad_reg)

        return F

    # ------------------------------------------------------------------
    # Feature extraction (for estimation bias computation)
    # ------------------------------------------------------------------

    def get_aggregated_features(self, A, X):
        """
        Extract propagated features after all QN-IRLS layers.

        Since RUNG propagates entirely in class space, the aggregated features
        ARE the final logits. This method is provided for compatibility with
        the estimation bias computation in utils/metrics.py.

        Reference: Section 4.3 and Figure 6 of the RUNG paper.

        Args:
            A: [N, N] adjacency matrix
            X: [N, D] node feature matrix

        Returns:
            F: [N, C] aggregated feature (= logit) matrix
        """
        return self.forward(A, X)

