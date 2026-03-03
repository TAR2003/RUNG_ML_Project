"""
model/rung_confidence_lambda.py

Extends RUNG_learnable_gamma by replacing the fixed scalar lambda
(skip connection weight) with a per-node confidence-weighted lambda
computed dynamically from the model's own MLP prediction confidence.

Model lineage:
    RUNG (base, NeurIPS 2024)
        └── RUNG_new_SCAD  (replaces MCP with SCAD penalty)
                └── RUNG_learnable_gamma (per-layer learnable SCAD gamma)
                        └── RUNG_confidence_lambda (THIS FILE)

Core idea:
    Adversarial edges are preferentially added to uncertain nodes.
    Those nodes receive the most corrupted neighbourhood signals but with
    a fixed scalar lambda every node has the same skip-connection strength.
    Per-node confidence-weighted lambda adapts to each node's reliability:
    the more (or less) confident a node's pre-aggregation MLP prediction,
    the more (or less) it should rely on its own features vs. the graph.

Architecture:
    1.  MLP: X → F^(0)         [N, C]  (class-space, same as RUNG_learnable_gamma)
    2.  confidence → λ_i       [N]     (from softmax(F^(0)))
    3.  K QN-IRLS layers with learnable per-layer SCAD gamma AND per-node λ
    4.  Return F^(K)            [N, C]  as final logits

    Since F^(0) = MLP(X) already lives in class-space (dim C), the
    pre-aggregation logits are available directly — no separate
    encoder/classifier split is needed, matching RUNG_learnable_gamma's
    architecture exactly.

New parameters vs RUNG_learnable_gamma:
    raw_alpha — learnable sharpness scalar (1 new parameter)
    Total: K + 1 parameters (K log_lam from learnable gamma, 1 raw_alpha)

Interface (identical to RUNG_learnable_gamma):
    forward(A, X)  A is dense [N, N] adjacency (no self-loops),
                   X is [N, D] node feature matrix.

Do NOT modify RUNG, RUNG_new_SCAD, or RUNG_learnable_gamma in any way.

Reference: "Robust Graph Neural Networks via Unbiased Aggregation" NeurIPS 2024
    Eq. 8 extended: F^(k+1) = (diag(q^(k)) + diag(λ))^{-1}
                               [(W^(k) ⊙ Ã)F^(k) + diag(λ)F^(0)]
    where λ = [λ_1,...,λ_N] is the per-node confidence-weighted vector.
"""

import numpy as np
import torch
import torch.nn.functional as nnF
from torch import nn

from model.mlp import MLP
from model.rung_learnable_gamma import scad_weight_differentiable
from utils import add_loops, pairwise_squared_euclidean, sym_norm


# ============================================================
#  CONFIDENCE-TO-LAMBDA MAPPING
# ============================================================

def compute_confidence_lambda(
    logits_0,
    lambda_base,
    alpha,
    mode='protect_uncertain',
    normalize=True,
    eps=1e-6,
):
    """
    Compute per-node lambda weights from initial MLP prediction confidence.

    Maps each node's prediction confidence to a skip-connection weight so
    that the QN-IRLS update uses node-specific lambda values rather than
    a single global scalar.

    Three modes implement different hypotheses about which nodes benefit
    from a stronger skip connection (higher lambda = more weight on F^(0)):

    Mode 'protect_uncertain':
        λ_i = λ_base * (1 - conf_i + ε)^α
        Uncertain nodes (low confidence) get HIGHER lambda.
        Hypothesis: uncertain nodes are attacked more; they should rely
        more on their own features to resist neighbourhood corruption.

    Mode 'protect_confident':
        λ_i = λ_base * conf_i^α
        Confident nodes get HIGHER lambda.
        Hypothesis: confident nodes have reliable self-features; they
        should insist on them even when neighbours are adversarial.

    Mode 'symmetric':
        λ_i = λ_base * (4 * conf_i * (1 - conf_i))^α
        Peak at conf=0.5; very uncertain AND very confident nodes both
        get low lambda.  Extreme predictions (either way) are less stable.

    After computing raw λ_i, optional normalization preserves the mean:
        λ_i = λ_i * (λ_base / mean(λ_i))
    This ensures total skip-connection strength equals RUNG_learnable_gamma,
    isolating the effect of redistribution from overall magnitude change.

    Args:
        logits_0:    Pre-aggregation MLP logits, shape [N, C].
        lambda_base: Base lambda scalar (Python float or scalar tensor).
        alpha:       Sharpness tensor with requires_grad=True.
                     Higher alpha = more extreme redistribution.
        mode:        One of 'protect_uncertain', 'protect_confident',
                     'symmetric'.
        normalize:   If True, normalise so mean(λ_i) = lambda_base.
        eps:         Small floor to prevent zero lambda or division edge cases.

    Returns:
        lambda_per_node: Per-node lambda values, shape [N].
                         Differentiable w.r.t. alpha and (indirectly) logits_0.
    """
    # Softmax probabilities and per-node max confidence
    probs = torch.softmax(logits_0, dim=-1)          # [N, C]
    conf  = probs.max(dim=-1).values                  # [N], in [1/C, 1.0]

    # Clamp to stable numerical range
    conf  = conf.clamp(min=eps, max=1.0 - eps)

    if mode == 'protect_uncertain':
        # Uncertainty = 1 - confidence  (high when the node is unsure)
        # Uncertain nodes get HIGHER lambda → more reliance on own features
        uncertainty = (1.0 - conf).clamp(min=eps)
        raw_lambda  = lambda_base * (uncertainty ** alpha)

    elif mode == 'protect_confident':
        # Confident nodes get HIGHER lambda → anchor on own features
        raw_lambda = lambda_base * (conf ** alpha)

    elif mode == 'symmetric':
        # 4 * conf * (1-conf) peaks at conf=0.5, decays to 0 at extremes
        symmetry   = (4.0 * conf * (1.0 - conf)).clamp(min=eps)
        raw_lambda = lambda_base * (symmetry ** alpha)

    else:
        raise ValueError(
            f"Unknown confidence mode: '{mode}'. "
            "Choose from: 'protect_uncertain', 'protect_confident', 'symmetric'."
        )

    # Normalize to preserve mean(lambda_i) = lambda_base
    # This isolates the redistribution effect from magnitude change,
    # making RUNG_confidence_lambda comparable to RUNG_learnable_gamma.
    if normalize:
        mean_lam        = raw_lambda.mean().clamp(min=eps)
        lambda_per_node = raw_lambda * (lambda_base / mean_lam)
    else:
        lambda_per_node = raw_lambda

    # Safety floor: no node should have exactly zero lambda
    lambda_per_node = lambda_per_node.clamp(min=eps)

    return lambda_per_node


# ============================================================
#  FULL MODEL: RUNG_confidence_lambda
# ============================================================

class RUNG_confidence_lambda(nn.Module):
    """
    RUNG with per-layer learnable SCAD gamma AND per-node confidence lambda.

    Extends RUNG_learnable_gamma with one additional mechanism:
    the skip-connection weight lambda is computed per-node from the
    model's own prediction confidence before graph aggregation.

    The per-node QN-IRLS update generalises Eq. 8 of the NeurIPS 2024 paper:

        F^(k+1) = (diag(q^(k)) + diag(λ))^{-1}
                  [(W^(k) ⊙ Ã) F^(k)  +  diag(λ) F^(0)]

    where λ = [λ_1,...,λ_N] is a VECTOR with
        λ_i = λ_base * g(conf_i, α, mode)

    Since diag(λ) is diagonal its inverse is trivially element-wise,
    so the per-node update costs exactly the same as the scalar case:
        f_i^(k+1) = [agg_i + λ_i * f_i^(0)] / (q_i + λ_i)

    New parameters vs RUNG_learnable_gamma:
        raw_alpha — 1 learnable scalar sharpness parameter.
    Total overhead: literally 1 extra float in the checkpoint.

    Args:
        in_dim:               Input feature dimension D.
        out_dim:              Number of output classes C.
        hidden_dims:          MLP hidden layer widths, e.g. [64].
        lam_hat:              Skip-connection fraction λ̂ ∈ (0, 1].
                              λ_base = 1/λ̂ − 1.
        gamma_init:           Initial SCAD zero-cutoff γ for all layers.
                              Internally: lam_init = gamma_init / scad_a.
        gamma_init_strategy:  'uniform', 'decreasing', or 'increasing'.
        scad_a:               SCAD shape parameter (default 3.7).
        prop_step:            Number of QN-IRLS iterations K (default 10).
        dropout:              MLP dropout rate.
        alpha_init:           Initial sharpness for confidence mapping.
                              alpha=1 → linear, alpha>1 → amplifies differences.
        confidence_mode:      'protect_uncertain' | 'protect_confident' | 'symmetric'.
        normalize_lambda:     If True, normalise mean(λ_i) = λ_base.
                              Recommended True for fair comparison.
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
        alpha_init: float = 1.0,
        confidence_mode: str = "protect_uncertain",
        normalize_lambda: bool = True,
    ):
        super().__init__()

        # ---- Validate inputs ----
        assert 0 < lam_hat <= 1, "lam_hat must be in (0, 1]"
        assert gamma_init > 0,   "gamma_init must be positive"
        assert scad_a > 1.0,     "SCAD shape param a must be > 1"
        assert confidence_mode in ('protect_uncertain', 'protect_confident', 'symmetric'), \
            f"Unknown confidence_mode: '{confidence_mode}'"

        # ---- Hyperparameters ----
        self.lam_hat            = lam_hat
        self.lam                = 1.0 / lam_hat - 1.0   # λ_base scalar (Python float)
        self.prop_layer_num     = prop_step
        self.scad_a             = scad_a
        self.gamma_init         = gamma_init
        self.gamma_init_strategy = gamma_init_strategy
        self.confidence_mode    = confidence_mode
        self.normalize_lambda   = normalize_lambda

        # ---- MLP backbone (identical to RUNG_learnable_gamma) ----
        # MLP maps raw features → class-space F^(0) [N, C].
        # Since RUNG aggregates in class space, F^(0) also serves directly
        # as pre-aggregation logits for confidence computation.
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # ---- Per-layer learnable SCAD thresholds (from RUNG_learnable_gamma) ----
        # log_lam^(k) is the learned parameter; lam^(k) = exp(log_lam^(k)).
        # lam_init = gamma_init / scad_a  (matches RUNG_learnable_gamma convention)
        gamma_initials = self._init_gammas(gamma_init, prop_step, gamma_init_strategy)
        lam_inits      = [g / scad_a for g in gamma_initials]

        self.log_lams = nn.ParameterList([
            nn.Parameter(
                torch.tensor(float(np.log(l)), dtype=torch.float32)
            )
            for l in lam_inits
        ])

        # ---- NEW: Learnable alpha (sharpness) ----
        # α = softplus(raw_alpha) + 0.5
        # Ensures α > 0.5 always, preventing degenerate near-zero alpha
        # (which would make all λ_i equal and deactivate the mechanism).
        # Initialise raw_alpha so that α ≈ alpha_init at the start:
        #   softplus(raw) + 0.5 = alpha_init
        #   softplus(raw) = alpha_init - 0.5
        #   raw = softplus_inverse(alpha_init - 0.5) = log(exp(max(alpha_init-0.5, 0.01)) - 1)
        alpha_floor = 0.5
        target      = max(alpha_init - alpha_floor, 0.01)
        raw_init    = float(np.log(max(float(np.exp(target) - 1), 1e-8)))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # ---- Config for logging / checkpointing ----
        self._config = {
            'model':                 'RUNG_confidence_lambda',
            'in_dim':                in_dim,
            'out_dim':               out_dim,
            'hidden_dims':           hidden_dims,
            'lam_hat':               lam_hat,
            'gamma_init':            gamma_init,
            'gamma_init_strategy':   gamma_init_strategy,
            'scad_a':                scad_a,
            'prop_step':             prop_step,
            'dropout':               dropout,
            'alpha_init':            alpha_init,
            'confidence_mode':       confidence_mode,
            'normalize_lambda':      normalize_lambda,
        }

    # ------------------------------------------------------------------
    #  Gamma initialisation strategy (identical to RUNG_learnable_gamma)
    # ------------------------------------------------------------------

    @staticmethod
    def _init_gammas(gamma_init: float, num_layers: int, strategy: str) -> list:
        """
        Return list of initial gamma values, one per layer.

        Strategies:
            'uniform':   all layers at gamma_init.
            'decreasing': gamma_k = gamma_init * exp(-k / (K-1))
            'increasing': gamma_k = gamma_init * exp(+k / (K-1))
        """
        K = num_layers
        if strategy == "uniform":
            return [gamma_init] * K
        elif strategy == "decreasing":
            return [gamma_init * float(np.exp(-k / max(K - 1, 1)))
                    for k in range(K)]
        elif strategy == "increasing":
            return [gamma_init * float(np.exp(k / max(K - 1, 1)))
                    for k in range(K)]
        else:
            raise ValueError(
                f"Unknown gamma_init_strategy: '{strategy}'. "
                "Choose from: uniform, decreasing, increasing."
            )

    # ------------------------------------------------------------------
    #  Alpha property
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Current alpha value as a Python float (for logging)."""
        return (nnF.softplus(self.raw_alpha) + 0.5).item()

    def get_alpha_tensor(self) -> torch.Tensor:
        """
        Return alpha as a differentiable tensor.

        Use inside forward() to preserve gradient through raw_alpha.
        """
        return nnF.softplus(self.raw_alpha) + 0.5

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with confidence-weighted per-node lambda.

        Flow:
            X → MLP → F^(0)  [N, C]
                        │
                        ├─► softmax → conf → λ_i  [N]   (per-node lambda)
                        │
                        └─► Layer 0: QN-IRLS(F^(0), W^(0), γ^(0), λ)
                                └─► Layer 1: QN-IRLS(F^(1), W^(1), γ^(1), λ)
                                        └─► ...
                                                └─► F^(K)  [N, C] logits

        Note: λ_i is computed ONCE from F^(0) and reused for all K layers.
        This is correct because λ_i reflects the reliability of F^(0) itself,
        which does not change during the aggregation loop.

        Args:
            A: [N, N] adjacency matrix (float, no self-loops expected).
            X: [N, D] node feature matrix.

        Returns:
            F: [N, C] logit matrix after K propagation iterations.
        """
        # ---- 1. MLP: raw features → class-space initial features F^(0) ----
        F0 = self.mlp(X)                              # [N, C]

        # ---- 2. Pre-aggregation confidence → per-node lambda ----
        # F^(0) serves directly as pre-aggregation logits (class-space output
        # of MLP, before any graph aggregation).  Confidence = max(softmax(F^(0))).
        alpha           = self.get_alpha_tensor()      # scalar tensor, grad ← raw_alpha
        lambda_per_node = compute_confidence_lambda(
            logits_0    = F0,
            lambda_base = self.lam,
            alpha       = alpha,
            mode        = self.confidence_mode,
            normalize   = self.normalize_lambda,
        )                                              # [N], differentiable

        # ---- 3. Graph preprocessing (identical to RUNG_learnable_gamma) ----
        A       = add_loops(A)                         # add self-loops
        D       = A.sum(-1)                            # [N] degree
        D_sq    = D.sqrt().unsqueeze(-1)               # [N, 1]
        A_tilde = sym_norm(A)                          # D^{-1/2} A D^{-1/2}

        F = F0

        # ---- 4. K QN-IRLS layers with learnable gamma + per-node lambda ----
        for log_lam_k in self.log_lams:

            # Learnable SCAD threshold for this layer (same as RUNG_learnable_gamma)
            lam_k = torch.exp(log_lam_k)               # scalar tensor, grad flows

            # y_{ij} = ||f_i/√d_i - f_j/√d_j||_2  (Eq. 8)
            # .detach() matches RUNG convention: IRLS treats y as fixed per outer iter
            Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
            y = Z.sqrt()

            # W_{ij} = dρ_SCAD(y_ij)/dy²  with learnable lam_k
            W = scad_weight_differentiable(y, lam_k, a=self.scad_a)

            # Zero diagonal — out-of-place keeps autograd intact for lam_k
            eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            W   = W * (1.0 - eye)

            # NaN guard for isolated nodes — out-of-place
            W = torch.where(torch.isnan(W), torch.ones_like(W), W)

            # Per-node QN-IRLS update (generalised Eq. 8 with vector lambda):
            #   q_i = Σ_j W_ij A_ij / d_i                         [N]
            #   f_i^(k+1) = [agg_i + λ_i * f_i^(0)] / (q_i + λ_i)
            #
            # Key difference from RUNG_learnable_gamma:
            #   scalar  self.lam    → vector  lambda_per_node  [N]
            # The inverse (diag(q) + diag(λ))^{-1} is still trivially element-wise.
            q     = (W * A).sum(-1) / D                # [N]
            q_hat = (q + lambda_per_node).unsqueeze(-1)  # [N, 1], clamp for safety
            q_hat = q_hat.clamp(min=1e-8)

            F = (W * A_tilde) @ F / q_hat + lambda_per_node.unsqueeze(-1) * F0 / q_hat

        return F

    # ------------------------------------------------------------------
    #  Feature extraction (estimation bias compatibility)
    # ------------------------------------------------------------------

    def get_aggregated_features(
        self, A: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """
        Return propagated features for estimation bias computation.

        Since RUNG_confidence_lambda aggregates in class space, the
        aggregated features ARE the final logits.  Provided for
        compatibility with utils/metrics.py.
        """
        return self.forward(A, X)

    # ------------------------------------------------------------------
    #  Optimizer helpers
    # ------------------------------------------------------------------

    def get_gamma_parameters(self):
        """Yield only the log_lam (gamma) parameters, one per layer."""
        return (p for p in self.log_lams.parameters())

    def get_alpha_parameters(self):
        """Yield only the raw_alpha parameter."""
        return iter([self.raw_alpha])

    def get_non_gamma_alpha_parameters(self):
        """Yield all parameters EXCEPT log_lam and raw_alpha."""
        special_ids = (
            {id(p) for p in self.log_lams.parameters()}
            | {id(self.raw_alpha)}
        )
        return (p for p in self.parameters() if id(p) not in special_ids)

    # Alias for compatibility with RUNG_learnable_gamma helper name
    def get_non_gamma_parameters(self):
        """Yield all parameters EXCEPT log_lam and raw_alpha (drop-in alias)."""
        return self.get_non_gamma_alpha_parameters()

    # ------------------------------------------------------------------
    #  Inspection / logging
    # ------------------------------------------------------------------

    def get_learned_gammas(self) -> list:
        """
        Return current learned zero-cutoff (= scad_a * lam) values per layer.

        Same scale as the CLI --gamma argument and RUNG_learnable_gamma,
        allowing direct comparison.
        """
        return [
            self.scad_a * float(torch.exp(log_lam).item())
            for log_lam in self.log_lams
        ]

    def get_learned_lams(self) -> list:
        """Return current lam^(k) = exp(log_lam^(k)) values per layer."""
        return [float(torch.exp(lp).item()) for lp in self.log_lams]

    def get_lambda_distribution(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
    ) -> tuple:
        """
        Return per-node lambda values and confidence scores for analysis.

        Useful for verifying that lambda redistribution is happening as
        expected: plotting lambda_i vs conf_i should show the mode-specific
        correlation pattern.

        Args:
            A: [N, N] adjacency matrix (clean or perturbed).
            X: [N, D] node features.

        Returns:
            lambda_vals: tensor [N], current per-node lambda values.
            conf_vals:   tensor [N], per-node prediction confidence.
            summary:     dict with statistics.
        """
        self.eval()
        with torch.no_grad():
            F0    = self.mlp(X)
            probs = torch.softmax(F0, dim=-1)
            conf  = probs.max(dim=-1).values
            alpha = self.get_alpha_tensor()

            lambda_vals = compute_confidence_lambda(
                logits_0    = F0,
                lambda_base = self.lam,
                alpha       = alpha,
                mode        = self.confidence_mode,
                normalize   = self.normalize_lambda,
            )

        summary = {
            'lambda_mean':  lambda_vals.mean().item(),
            'lambda_std':   lambda_vals.std().item(),
            'lambda_min':   lambda_vals.min().item(),
            'lambda_max':   lambda_vals.max().item(),
            'lambda_base':  self.lam,
            'conf_mean':    conf.mean().item(),
            'conf_std':     conf.std().item(),
            'alpha':        alpha.item(),
            'mode':         self.confidence_mode,
        }

        return lambda_vals.cpu(), conf.cpu(), summary

    def log_gamma_stats(self) -> None:
        """
        Print learned gamma / lam values per layer plus alpha.

        Output format matches RUNG_learnable_gamma.log_gamma_stats()
        for easy side-by-side comparison.
        """
        print(f"\n{'=' * 58}")
        print(f"{'RUNG_confidence_lambda — Gamma + Alpha Stats':^58}")
        print(f"{'=' * 58}")
        print(f"  confidence_mode:  {self.confidence_mode}")
        print(f"  normalize_lambda: {self.normalize_lambda}")
        print(f"  alpha:            {self.alpha:.4f}")
        print(f"  lambda_base:      {self.lam:.4f}")
        print()
        print(f"  {'Layer':>7}  {'lam':>8}  {'gamma(=a*lam)':>14}  {'log_lam':>9}")
        print(f"  {'-' * 48}")
        for k, log_lam_param in enumerate(self.log_lams):
            ll  = log_lam_param.item()
            lam = float(np.exp(ll))
            g   = self.scad_a * lam
            print(f"  {k:>7}    {lam:>8.4f}    {g:>14.4f}    {ll:>9.4f}")
        print(f"{'=' * 58}\n")

    def __repr__(self) -> str:
        gammas = self.get_learned_gammas()
        return (
            f"RUNG_confidence_lambda(\n"
            f"  prop_step={self.prop_layer_num},\n"
            f"  lam_hat={self.lam_hat},\n"
            f"  confidence_mode='{self.confidence_mode}',\n"
            f"  normalize_lambda={self.normalize_lambda},\n"
            f"  alpha={self.alpha:.3f},\n"
            f"  gamma_init={self.gamma_init},\n"
            f"  gamma_init_strategy='{self.gamma_init_strategy}',\n"
            f"  gammas_current=[{', '.join(f'{g:.3f}' for g in gammas)}],\n"
            f"  scad_a={self.scad_a}\n"
            f")"
        )
