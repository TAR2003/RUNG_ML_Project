"""
Penalty functions for robust graph signal estimation in RUNG.

For IRLS (Iteratively Reweighted Least Squares) we need the derivative of the
penalty ρ(y) with respect to y², evaluated at the current edge feature
differences y = ||f_i/√d_i - f_j/√d_j||_2.

This gives the edge reweighting:   W_ij = dρ(y_ij) / dy²

All functions operate on dense [N, N] matrices (the codebase uses dense adj).
All functions return W_ij >= 0, with W_ij = 0 meaning "prune this edge".

Reference: Section 3.1 and Eq. 8 of RUNG paper (NeurIPS 2024)
"""

import torch


class PenaltyFunction:
    """
    Collection of penalty functions ρ(y) for robust graph signal estimation.

    All static methods follow the same interface:
        y : [N, N] pairwise feature difference tensor (y_ij >= 0)
        returns W : [N, N] edge weights (W_ij >= 0)

    The factory method `get_w_func` returns a callable compatible with the
    RUNG model's `w_func` interface.
    """

    # ------------------------------------------------------------------
    # Core penalty derivative functions
    # ------------------------------------------------------------------

    @staticmethod
    def mcp(y: torch.Tensor, gamma: float, eps: float = 1e-8) -> torch.Tensor:
        """
        Minimax Concave Penalty (MCP) derivative for IRLS edge reweighting.

        dρ_gamma(y)/dy² = max(0, 1/(2y) - 1/(2*gamma))

        Properties:
        - Equals l1 derivative (1/2y) for small y (y ≪ gamma)
        - Becomes exactly ZERO for y >= gamma  →  zero estimation bias
        - Controlled by single threshold gamma (larger gamma ≈ closer to l1)

        This is the default penalty used in the RUNG paper.

        Args:
            y:     Pairwise feature diffs, shape [N, N], values >= 0
            gamma: MCP threshold (larger = closer to l1, less pruning)
            eps:   Clamp floor for numerical stability (avoids 1/0)

        Returns:
            W:     Edge weights, shape [N, N], values in [0, 1/(2*eps)]
        """
        y_safe = y.clamp(min=eps)
        W = torch.clamp(1.0 / (2.0 * y_safe) - 1.0 / (2.0 * gamma), min=0.0)
        return W

    @staticmethod
    def scad(
        y: torch.Tensor, lam: float, a: float = 3.7, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Smoothly Clipped Absolute Deviation (SCAD) derivative for IRLS.

        Proposed by Fan & Li (2001) for variable selection. Has three regions:
          - y < lam:           full l1 weight 1/(2y)
          - lam <= y < a*lam:  linearly decreasing toward zero
          - y >= a*lam:        exactly zero (same zero-bias property as MCP)

        dρ_SCAD(y)/dy² =
            1/(2y)                               if y < lam
            (a*lam - y) / ((a-1)*lam * 2*y)     if lam <= y < a*lam
            0                                    if y >= a*lam

        Args:
            y:    Pairwise feature diffs, shape [N, N]
            lam:  SCAD threshold λ (analogous to gamma in MCP)
            a:    SCAD shape parameter, default 3.7 (Fan & Li standard value)
            eps:  Numerical stability floor

        Returns:
            W:    Edge weights, shape [N, N]
        """
        y_safe = y.clamp(min=eps)

        l1_weight = 1.0 / (2.0 * y_safe)
        transition_weight = (a * lam - y) / ((a - 1) * lam * 2.0 * y_safe)
        transition_weight = transition_weight.clamp(min=0.0)

        # Region masks (mutually exclusive, region 3 = default zero)
        region1 = y < lam
        region2 = (y >= lam) & (y < a * lam)

        W = torch.zeros_like(y)
        W[region1] = l1_weight[region1]
        W[region2] = transition_weight[region2]
        return W

    @staticmethod
    def l1(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        L1 penalty derivative for IRLS (biased baseline).

        dρ_l1(y)/dy² = 1/(2y)

        This is the standard geometric median estimator, shown in the RUNG paper
        (Figure 2 and Section 2.3) to have O(budget) estimation bias.

        Args:
            y:   Pairwise feature diffs, shape [N, N]
            eps: Numerical stability floor

        Returns:
            W:   Edge weights, shape [N, N]
        """
        return 1.0 / (2.0 * y.clamp(min=eps))

    @staticmethod
    def l2(y: torch.Tensor) -> torch.Tensor:
        """
        L2 penalty derivative for IRLS (reduces to GCN / APPNP).

        dρ_l2(y)/dy² = 1 (constant, independent of y)

        When all W_ij = 1, Eq. 8 reduces to standard APPNP propagation.
        Use to verify that removing robustness degrades to GCN performance.

        Args:
            y:   Pairwise feature diffs (unused, kept for interface consistency)

        Returns:
            W:   All-ones tensor, shape same as y
        """
        return torch.ones_like(y)

    @staticmethod
    def homophily_adaptive(
        y: torch.Tensor,
        gamma: float,
        soft_labels: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Homophily-aware adaptive edge weight for heterophilic graphs.

        Motivation: RUNG's MCP assumes large feature differences signal
        adversarial edges. On heterophilic graphs this is wrong — different-class
        nodes are legitimately connected and have large feature differences.
        This weight adapts based on predicted class similarity.

        Math:
            h_ij = 1 - Σ_c(p_ic · p_jc)        [heterophily: 0=homo, 1=hetero]
            W_homo(y)   = max(0, 1/2y - 1/2γ)   [MCP: suppress dissimilar features]
            W_hetero(y) = max(0, 1/2γ - 1/2y)   [inverted: suppress similar features]
            W_adaptive  = (1 - h_ij)*W_homo + h_ij*W_hetero

        Properties:
        - h_ij = 0 (homophilic edge): reduces to standard MCP (same as RUNG)
        - h_ij = 1 (heterophilic edge): inverted weight (preserves cross-class edges)
        - Smoothly interpolates between regimes based on soft class predictions

        Args:
            y:           Pairwise feature diffs, shape [N, N]
            gamma:       MCP threshold parameter
            soft_labels: Softmax class probs, shape [N, C]
            eps:         Numerical stability floor

        Returns:
            W:           Adaptive edge weights, shape [N, N]
        """
        # h_ij = 1 - Σ_c(p_ic · p_jc)  (1 - inner product of prob vectors)
        # dot_product[i, j] = Σ_c p_ic * p_jc  (high for same-class nodes)
        dot_product = soft_labels @ soft_labels.t()          # [N, N]
        h = 1.0 - dot_product.clamp(0.0, 1.0)               # [N, N] heterophily

        y_safe = y.clamp(min=eps)
        W_homo   = torch.clamp(1.0 / (2.0 * y_safe) - 1.0 / (2.0 * gamma), min=0.0)
        W_hetero = torch.clamp(1.0 / (2.0 * gamma) - 1.0 / (2.0 * y_safe), min=0.0)

        W_adaptive = (1.0 - h) * W_homo + h * W_hetero
        return W_adaptive.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Factory: returns a callable compatible with RUNG's w_func interface
    # ------------------------------------------------------------------

    @staticmethod
    def get_w_func(penalty: str, gamma: float = 3.0, a: float = 3.7):
        """
        Return a callable  w_func(y) -> W  for use as RUNG's edge weight function.

        The returned callable takes a [N, N] tensor y (pairwise L2 diffs) and
        returns a [N, N] tensor W (edge weights).

        Note: 'adaptive' cannot be used here because it requires soft_labels at
        runtime. Use penalty='adaptive' directly in RUNG.__init__ instead.

        Args:
            penalty: One of 'mcp', 'scad', 'l1', 'l2'
            gamma:   Threshold parameter (MCP gamma or SCAD lam)
            a:       SCAD shape parameter (default 3.7, only used for SCAD)

        Returns:
            Callable w_func(y: Tensor) -> Tensor
        """
        if penalty == 'mcp':
            return lambda y: PenaltyFunction.mcp(y, gamma)
        elif penalty == 'scad':
            return lambda y: PenaltyFunction.scad(y, lam=gamma, a=a)
        elif penalty == 'l1':
            return lambda y: PenaltyFunction.l1(y)
        elif penalty == 'l2':
            return lambda y: PenaltyFunction.l2(y)
        else:
            raise ValueError(
                f"Unknown penalty: '{penalty}'. "
                f"Choose from ['mcp', 'scad', 'l1', 'l2']. "
                f"For 'adaptive', pass penalty='adaptive' to RUNG.__init__."
            )
