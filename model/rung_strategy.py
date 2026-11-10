"""Composable RUNG used by the strategy presets in ``model/strategy_presets.py``.

Each preset in the roadmap (S01-S34) is a setting of the switches below.
The QN-IRLS update matches ``model/rung.py``.

Switches
    distance:     euclidean | cosine | hidden_cosine
    gamma_mode:   fixed | percentile | parametric | blend | banded
                  | bounded_schedule | bounded_per_layer | per_node_q | per_node
    penalty:      mcp | scad | adaptive | same_class | two_channel
    lambda_mode:  scalar | confidence | layer
    readout:      last | jump
    edge_only:    aggregate from edges instead of a dense N x N weight matrix
    early_exit:   stop propagation when the state stops moving
    fp16_distance: compute the pairwise score in float16 on CUDA
    detach_distance: standard IRLS detach; the trainer can turn it off
                     for the first ``detach_warmup_epochs`` epochs
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from model.rung_learnable_gamma import scad_weight_differentiable
from utils import add_loops, sym_norm


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


class MLPWithHidden(nn.Module):
    """Same MLP as ``model.mlp.MLP``, plus an optional hidden activation."""

    def __init__(self, n_feat, n_class, hidden_dims, bias=True, dropout=0.5):
        super().__init__()
        dims = list(hidden_dims)
        self.linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias)
            for in_dim, out_dim in zip([n_feat] + dims, dims + [n_class])
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, return_hidden=False):
        h = X
        for linear in self.linears[:-1]:
            h = self.dropout(F.relu(linear(h)))
        logits = self.linears[-1](h)
        if return_hidden:
            return logits, h
        return logits


class RUNGStrategy(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims=None,
        lam_hat=0.9,
        prop_step=10,
        dropout=0.5,
        distance="euclidean",
        gamma_mode="fixed",
        penalty="mcp",
        gamma=6.0,
        gamma_bound=None,
        percentile_q=0.75,
        decay_rate_init=0.85,
        scad_a=3.7,
        q_relax=0.0,
        q_max=0.99,
        homophily_mode="from_F0",
        lambda_mode="scalar",
        lambda_schedule=1.0,
        readout="last",
        edge_only=False,
        early_exit=False,
        early_exit_tol=1e-3,
        fp16_distance=False,
        detach_distance=True,
        detach_warmup_epochs=0,
        two_channel_beta=0.5,
        cross_class_mode="keep",
        cross_class_alpha=0.1,
        band_lo=0.5,
        band_hi=0.95,
        eps=1e-8,
        strategy_id="custom",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64]

        assert 0 < lam_hat <= 1
        assert distance in ("euclidean", "cosine", "hidden_cosine")
        assert gamma_mode in (
            "fixed", "percentile", "parametric", "blend", "banded",
            "bounded_schedule", "bounded_per_layer", "per_node_q", "per_node",
        )
        assert penalty in ("mcp", "scad", "adaptive", "same_class", "two_channel")
        assert lambda_mode in ("scalar", "confidence", "layer")
        assert readout in ("last", "jump")
        assert cross_class_mode in ("keep", "negative")
        assert 0.0 < percentile_q < 1.0
        assert scad_a > 1.0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lam_hat = float(lam_hat)
        self.lam = 1.0 / float(lam_hat) - 1.0
        self.prop_step = int(prop_step)
        self.prop_layer_num = self.prop_step
        self.distance = distance
        self.gamma_mode = gamma_mode
        self.penalty = penalty
        self.gamma = float(gamma)
        self.gamma_bound = None if gamma_bound is None else float(gamma_bound)
        self.percentile_q = float(percentile_q)
        self.decay_rate_init = float(decay_rate_init)
        self.scad_a = float(scad_a)
        self.q_relax = float(q_relax)
        self.q_max = float(q_max)
        self.homophily_mode = homophily_mode
        self.lambda_mode = lambda_mode
        self.lambda_schedule = float(lambda_schedule)
        self.readout = readout
        self.edge_only = bool(edge_only)
        self.early_exit = bool(early_exit)
        self.early_exit_tol = float(early_exit_tol)
        self.fp16_distance = bool(fp16_distance)
        self.detach_distance = bool(detach_distance)
        self.detach_warmup_epochs = int(detach_warmup_epochs)
        self.two_channel_beta = float(two_channel_beta)
        self.cross_class_mode = cross_class_mode
        self.cross_class_alpha = float(cross_class_alpha)
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        self.eps = float(eps)
        self.strategy_id = strategy_id

        self.train_mode = "clean"
        self.report_bias = False
        self.consistency_weight = 0.1
        self.current_epoch = 10 ** 9
        self.last_exit_layer = self.prop_step
        self._last_gammas = [None] * self.prop_step
        self.last_transition_frac = 0.0

        self.mlp = MLPWithHidden(in_dim, out_dim, hidden_dims, dropout=dropout)
        self._init_schedules()
        if readout == "jump":
            self.jump_logits = nn.Parameter(torch.zeros(3))

    def _init_schedules(self):
        bound = self.gamma_bound
        use_bound = bound is not None or self.gamma_mode in (
            "bounded_schedule", "bounded_per_layer",
        )
        if self.gamma_mode in ("parametric", "blend", "banded", "per_node", "bounded_schedule"):
            if use_bound or self.distance in ("cosine", "hidden_cosine"):
                cap = bound if bound is not None else 2.0
                frac = min(max(self.gamma / cap, 1e-3), 1.0 - 1e-3)
                self.raw_g0 = nn.Parameter(torch.tensor(_logit(frac)))
                self._gamma_cap = cap
            else:
                self.raw_g0 = nn.Parameter(torch.tensor(math.log(max(self.gamma, self.eps))))
                self._gamma_cap = None
            self.raw_decay = nn.Parameter(torch.tensor(_logit(self.decay_rate_init)))
        if self.gamma_mode == "blend":
            self.raw_blend = nn.Parameter(torch.tensor(0.0))
        if self.gamma_mode == "bounded_per_layer":
            cap = bound if bound is not None else 2.0
            self._gamma_cap = cap
            self.raw_gammas = nn.Parameter(torch.zeros(self.prop_step))
        if self.gamma_mode == "per_node":
            self.raw_c = nn.Parameter(torch.tensor(0.0))
        if self.lambda_mode == "confidence":
            self.raw_sharp = nn.Parameter(torch.tensor(0.0))

    def get_last_gammas(self):
        return list(self._last_gammas)

    def get_aggregated_features(self, A, X):
        return self.forward(A, X)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gamma_parameters(self):
        names = ("raw_g0", "raw_decay", "raw_blend", "raw_gammas", "raw_c", "raw_sharp", "jump_logits")
        params = []
        for name in names:
            if hasattr(self, name):
                params.append(getattr(self, name))
        return params

    def build_optimizer(self, lr=5e-2, weight_decay=5e-4, gamma_lr_factor=0.3):
        gamma_ids = {id(p) for p in self.gamma_parameters()}
        main = [p for p in self.parameters() if id(p) not in gamma_ids and p.requires_grad]
        gamma = [p for p in self.gamma_parameters() if p.requires_grad]
        groups = [{"params": main, "lr": lr, "weight_decay": weight_decay}]
        if gamma:
            groups.append({
                "params": gamma,
                "lr": lr * gamma_lr_factor,
                "weight_decay": 0.0,
            })
        return torch.optim.Adam(groups)

    def _should_detach(self):
        if not self.detach_distance:
            return False
        return self.current_epoch >= self.detach_warmup_epochs

    def _schedule_gamma(self, layer_idx):
        if self.gamma_mode == "bounded_per_layer":
            return torch.sigmoid(self.raw_gammas[layer_idx]) * self._gamma_cap
        g0 = self.raw_g0
        if getattr(self, "_gamma_cap", None) is not None and self.gamma_mode != "fixed":
            g0 = torch.sigmoid(self.raw_g0) * self._gamma_cap
        elif self.gamma_mode in ("parametric", "blend", "banded", "per_node", "bounded_schedule"):
            if getattr(self, "_gamma_cap", None) is None:
                g0 = torch.exp(self.raw_g0)
        decay = torch.sigmoid(self.raw_decay)
        return g0 * torch.pow(decay, layer_idx)

    def _edges(self, A):
        idx = (A > 0).nonzero(as_tuple=False)
        if idx.numel() == 0:
            empty = idx.new_zeros((0,))
            return empty, empty
        keep = idx[:, 0] != idx[:, 1]
        idx = idx[keep]
        return idx[:, 0], idx[:, 1]

    def _pair_distance(self, features, src, dst):
        use_fp16 = self.fp16_distance and features.is_cuda
        feats = features
        if self.distance == "euclidean":
            if use_fp16:
                feats = features.half()
            if src is None:
                sq = (feats * feats).sum(-1)
                y2 = sq[:, None] + sq[None, :] - 2.0 * (feats @ feats.transpose(-2, -1))
                y = y2.clamp(min=0).sqrt()
            else:
                y = (feats[src] - feats[dst]).norm(dim=-1)
            return y.float() if use_fp16 else y

        denom = feats.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        unit = feats / denom
        if use_fp16:
            unit = unit.half()
        if src is None:
            y = 1.0 - (unit @ unit.transpose(-2, -1))
            y = y.clamp(0.0, 2.0)
        else:
            y = (1.0 - (unit[src] * unit[dst]).sum(-1)).clamp(0.0, 2.0)
        return y.float() if use_fp16 else y

    def _edge_values(self, dense, A, src, dst):
        if src is not None:
            return dense
        mask = (A > 0)
        eye = torch.eye(A.shape[0], device=A.device, dtype=torch.bool)
        mask = mask & ~eye
        if int(mask.sum()) == 0:
            return dense.new_zeros((0,))
        return dense[mask]

    def _quantile(self, values, q):
        if values.numel() == 0:
            return values.new_tensor(self.gamma)
        q_t = torch.as_tensor(q, device=values.device, dtype=values.dtype)
        return torch.quantile(values, q_t).clamp(min=self.eps)

    def _node_homophily(self, logits, A):
        probs = torch.softmax(logits.detach(), dim=-1)
        agree = probs @ probs.transpose(-2, -1)
        eye = torch.eye(A.shape[0], device=A.device, dtype=torch.bool)
        mask = (A > 0) & ~eye
        deg = mask.sum(-1).clamp(min=1).to(agree.dtype)
        h = (agree * mask.to(agree.dtype)).sum(-1) / deg
        missing = mask.sum(-1) == 0
        floor = 1.0 / float(self.out_dim)
        h = torch.where(missing, h.new_full(h.shape, floor), h)
        return h.clamp(floor, 1.0)

    def _gamma_for_layer(self, y, layer_idx, logits, A, src, dst):
        edges = y if src is not None else self._edge_values(y, A, src, dst)
        mode = self.gamma_mode

        if mode == "fixed":
            return y.new_tensor(self.gamma)

        if mode == "percentile":
            return self._quantile(edges, self.percentile_q)

        if mode in ("parametric", "bounded_schedule"):
            return self._schedule_gamma(layer_idx)

        if mode == "bounded_per_layer":
            return self._schedule_gamma(layer_idx)

        if mode == "blend":
            prior = self._schedule_gamma(layer_idx)
            data = self._quantile(edges, self.percentile_q)
            alpha = torch.sigmoid(self.raw_blend)
            return alpha * prior + (1.0 - alpha) * data

        if mode == "banded":
            prior = self._schedule_gamma(layer_idx)
            lo = self._quantile(edges, self.band_lo)
            hi = self._quantile(edges, self.band_hi)
            return prior.clamp(min=lo, max=hi)

        h = self._node_homophily(logits, A)
        if mode == "per_node":
            prior = self._schedule_gamma(layer_idx)
            scale = 1.0 + F.softplus(self.raw_c) * (1.0 - h)
            g_node = prior * scale
            return self._broadcast_nodes(g_node, y, src, dst)

        q = (self.percentile_q + (1.0 - h) * self.q_relax).clamp(self.percentile_q, self.q_max)
        return self._per_node_quantile(y, A, q, src, dst)

    def _broadcast_nodes(self, g_node, y, src, dst):
        if src is None:
            return torch.maximum(g_node[:, None], g_node[None, :])
        return torch.maximum(g_node[src], g_node[dst])

    def _per_node_quantile(self, y, A, q, src, dst):
        if src is not None:
            # Edge storage has no row layout. Scale the global quantile by q_i / q_base.
            base = self._quantile(y, self.percentile_q)
            scale = (q / self.percentile_q).clamp(min=self.eps)
            g_node = base * scale
            return self._broadcast_nodes(g_node, y, src, dst)

        n = y.shape[0]
        eye = torch.eye(n, device=y.device, dtype=torch.bool)
        edge = (A > 0) & ~eye
        filled = torch.where(edge, y, y.new_full(y.shape, 1e6))
        order = filled.sort(dim=-1).values
        deg = edge.sum(-1).clamp(min=1)
        idx = torch.floor(q * (deg.to(q.dtype) - 1.0)).long().clamp(min=0, max=n - 1)
        gammas = order.gather(1, idx.unsqueeze(1)).squeeze(1)
        fallback = self._quantile(self._edge_values(y, A, None, None), self.percentile_q)
        gammas = torch.where(edge.any(-1), gammas, fallback)
        gammas = gammas.clamp(min=self.eps)
        return self._broadcast_nodes(gammas, y, src, dst)

    def _mcp(self, y, gamma):
        y_safe = y.clamp(min=self.eps)
        g = gamma.clamp(min=self.eps)
        return (1.0 / (2.0 * y_safe) - 1.0 / (2.0 * g)).clamp(min=0.0)

    def _base_weights(self, y, gamma):
        if self.penalty in ("mcp", "adaptive", "same_class", "two_channel"):
            return self._mcp(y, gamma)
        lam = gamma.clamp(min=self.eps) / self.scad_a
        return scad_weight_differentiable(y, lam, a=self.scad_a, eps=self.eps)

    def _record_transition(self, y, gamma):
        lam = gamma.clamp(min=self.eps) / self.scad_a
        band = (y >= lam) & (y < self.scad_a * lam)
        self.last_transition_frac = float(band.float().mean().detach()) if y.numel() else 0.0

    def _weights(self, y, gamma, logits, A, src, dst):
        self._record_transition(y if src is not None else self._edge_values(y, A, src, dst), gamma if src is not None or gamma.ndim == 0 else self._edge_values(gamma, A, src, dst))
        if self.penalty == "two_channel":
            return self._base_weights(y, gamma)

        base = self._base_weights(y, gamma)
        probs = torch.softmax(logits.detach(), dim=-1)
        if self.penalty == "adaptive":
            if src is None:
                het = 1.0 - (probs @ probs.transpose(-2, -1)).clamp(0.0, 1.0)
            else:
                het = 1.0 - (probs[src] * probs[dst]).sum(-1).clamp(0.0, 1.0)
            homo = self._mcp(y, gamma if gamma.ndim == 0 or gamma.shape == y.shape else gamma)
            # Inverted MCP keeps large gaps, which are legitimate on heterophilic edges.
            y_safe = y.clamp(min=self.eps)
            g = gamma.clamp(min=self.eps)
            hetero = (1.0 / (2.0 * g) - 1.0 / (2.0 * y_safe)).clamp(min=0.0)
            return (1.0 - het) * homo + het * hetero

        if self.penalty == "same_class":
            labels = probs.argmax(dim=-1)
            if src is None:
                same = labels[:, None] == labels[None, :]
            else:
                same = labels[src] == labels[dst]
            if self.cross_class_mode == "keep":
                other = torch.ones_like(base)
            else:
                other = base.new_full(base.shape, -self.cross_class_alpha)
            return torch.where(same, base, other)
        return base

    def _lambda_vector(self, logits, n):
        if self.lambda_mode != "confidence":
            return logits.new_tensor(self.lam)
        conf = torch.softmax(logits.detach(), dim=-1).max(dim=-1).values.clamp(min=self.eps)
        sharp = F.softplus(self.raw_sharp) + 0.1
        scale = conf.pow(-sharp)
        scale = scale / scale.mean().clamp(min=self.eps)
        return self.lam * scale

    def _qn(self, F, F0, W, A, A_tilde, D, lam, src, dst):
        if lam.ndim == 0:
            lam_v = lam
        else:
            lam_v = lam
        if src is None:
            eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            W = W * (1.0 - eye)
            W = torch.where(torch.isnan(W), torch.ones_like(W), W)
            denom_w = W.abs() if self.cross_class_mode == "negative" else W
            row = (denom_w * A).sum(-1) / D
            if lam_v.ndim == 0:
                Q = (row + lam_v).unsqueeze(-1).clamp(min=self.eps)
                skip = lam_v * F0
            else:
                Q = (row + lam_v).unsqueeze(-1).clamp(min=self.eps)
                skip = lam_v.unsqueeze(-1) * F0
            return (W * A_tilde) @ F / Q + skip / Q

        a_ij = A[src, dst]
        di = D[src].clamp(min=self.eps).sqrt()
        dj = D[dst].clamp(min=self.eps).sqrt()
        at = a_ij / (di * dj)
        w = torch.where(torch.isnan(W), torch.ones_like(W), W)
        denom_w = w.abs() if self.cross_class_mode == "negative" else w
        agg = torch.zeros_like(F)
        if src.numel():
            agg.index_add_(0, src, (w * at).unsqueeze(-1) * F[dst])
        row = torch.zeros(F.shape[0], device=F.device, dtype=F.dtype)
        if src.numel():
            row.index_add_(0, src, denom_w * a_ij)
        row = row / D
        if lam_v.ndim == 0:
            Q = (row + lam_v).unsqueeze(-1).clamp(min=self.eps)
            skip = lam_v * F0
        else:
            Q = (row + lam_v).unsqueeze(-1).clamp(min=self.eps)
            skip = lam_v.unsqueeze(-1) * F0
        return agg / Q + skip / Q

    def _propagate_once(self, F, F0, H, A, A_tilde, D, D_sq, lam_base, layer_idx, src, dst):
        if self.distance == "hidden_cosine":
            features = H / D_sq
        else:
            features = F / D_sq
        y = self._pair_distance(features, src, dst)
        if self._should_detach():
            y = y.detach()
        ref = F0 if self.homophily_mode == "from_F0" else F
        gamma = self._gamma_for_layer(y, layer_idx, ref, A, src, dst)
        stored = gamma.detach()
        self._last_gammas[layer_idx] = float(stored.mean()) if stored.ndim > 0 else float(stored)

        lam = lam_base
        if self.lambda_mode == "layer":
            lam = lam_base * (self.lambda_schedule ** layer_idx)

        if self.penalty == "two_channel":
            labels = torch.softmax(F.detach(), dim=-1).argmax(dim=-1)
            if src is None:
                same = labels[:, None] == labels[None, :]
            else:
                same = labels[src] == labels[dst]
            base = self._base_weights(y, gamma)
            w_lp = base * same.to(base.dtype)
            w_hp = (~same).to(base.dtype)
            beta = self.two_channel_beta
            low = self._qn(F, F0, w_lp, A, A_tilde, D, lam, src, dst)
            high = self._qn(F, F0, w_hp, A, A_tilde, D, lam, src, dst)
            return (1.0 - beta) * low + beta * high

        W = self._weights(y, gamma, ref, A, src, dst)
        return self._qn(F, F0, W, A, A_tilde, D, lam, src, dst)

    def forward(self, A, X):
        F0, H = self.mlp(X, return_hidden=True)
        A = add_loops(A)
        D = A.sum(-1).clamp(min=self.eps)
        D_sq = D.sqrt().unsqueeze(-1)
        A_tilde = sym_norm(A)
        src = dst = None
        if self.edge_only:
            src, dst = self._edges(A)

        lam_base = self._lambda_vector(F0, F0.shape[0])
        F = F0
        saved = [F0]
        mid = max(self.prop_step // 2 - 1, 0)
        exited = 0
        for k in range(self.prop_step):
            nxt = self._propagate_once(
                F, F0, H, A, A_tilde, D, D_sq, lam_base, k, src, dst,
            )
            exited = k + 1
            if self.early_exit and k > 0:
                delta = (nxt - F).norm() / F.norm().clamp(min=self.eps)
                F = nxt
                if float(delta.detach()) < self.early_exit_tol:
                    break
            else:
                F = nxt
            if k == mid:
                saved.append(F)
        self.last_exit_layer = exited
        saved.append(F)
        if self.readout == "jump":
            while len(saved) < 3:
                saved.insert(1, saved[-1])
            alpha = torch.softmax(self.jump_logits, dim=0)
            return alpha[0] * saved[0] + alpha[1] * saved[1] + alpha[2] * saved[-1]
        return F

    def log_stats(self):
        gammas = [g for g in self._last_gammas if g is not None]
        text = ", ".join(f"{g:.3f}" for g in gammas) if gammas else "n/a"
        print(
            f"  strategy={self.strategy_id} distance={self.distance} "
            f"gamma_mode={self.gamma_mode} penalty={self.penalty} "
            f"exit={self.last_exit_layer}/{self.prop_step} "
            f"transition={self.last_transition_frac:.3f} gammas=[{text}]"
        )


def _labeled_edge_homophily(A, y, train_idx):
    n = A.shape[0]
    train = torch.zeros(n, dtype=torch.bool, device=A.device)
    train[train_idx] = True
    src, dst = (A > 0).nonzero(as_tuple=False).T
    if src.numel() == 0:
        return 0.5
    keep = (src != dst) & train[src] & train[dst]
    if int(keep.sum()) == 0:
        keep = src != dst
    if int(keep.sum()) == 0:
        return 0.5
    return float((y[src[keep]] == y[dst[keep]]).float().mean())


class RUNGRouter(nn.Module):
    """S34. Pick one expert from training-label homophily and graph size.

    High homophily uses cosine + parametric decay.
    Low homophily on a small graph uses Euclidean parametric decay.
    Low homophily on a larger graph uses cosine + percentile gamma.
    """

    def __init__(self, in_dim, out_dim, hidden_dims=None, homophily_threshold=0.5, **shared):
        super().__init__()
        self.homophily_threshold = float(homophily_threshold)
        self.strategy_id = "S34"
        self.train_mode = "clean"
        self.report_bias = False
        self.consistency_weight = 0.1
        common = dict(shared)
        common["in_dim"] = in_dim
        common["out_dim"] = out_dim
        if hidden_dims is not None:
            common["hidden_dims"] = hidden_dims
        self.experts = nn.ModuleDict({
            "homo": RUNGStrategy(
                distance="cosine", gamma_mode="parametric", penalty="scad",
                gamma=1.0, gamma_bound=2.0, strategy_id="S34_homo", **_router_kwargs(common),
            ),
            "webkb": RUNGStrategy(
                distance="euclidean", gamma_mode="parametric", penalty="scad",
                gamma=3.0, gamma_bound=None, strategy_id="S34_webkb", **_router_kwargs(common),
            ),
            "wiki": RUNGStrategy(
                distance="cosine", gamma_mode="percentile", penalty="scad",
                percentile_q=0.75, strategy_id="S34_wiki", **_router_kwargs(common),
            ),
        })
        self.register_buffer("route_id", torch.zeros((), dtype=torch.long))
        self.last_homophily = None

    def set_graph_stats(self, A, y, train_idx):
        h = _labeled_edge_homophily(A, y, train_idx)
        self.last_homophily = h
        if h >= self.homophily_threshold:
            rid = 0
        elif A.shape[0] < 800:
            rid = 1
        else:
            rid = 2
        self.route_id.fill_(rid)

    def active_expert(self):
        return self.experts[("homo", "webkb", "wiki")[int(self.route_id.item())]]

    def gamma_parameters(self):
        return self.active_expert().gamma_parameters()

    def build_optimizer(self, lr=5e-2, weight_decay=5e-4, gamma_lr_factor=0.3):
        return self.active_expert().build_optimizer(lr, weight_decay, gamma_lr_factor)

    def forward(self, A, X):
        return self.active_expert()(A, X)

    def get_aggregated_features(self, A, X):
        return self.forward(A, X)

    def get_last_gammas(self):
        return self.active_expert().get_last_gammas()

    def log_stats(self):
        name = ("homo", "webkb", "wiki")[int(self.route_id.item())]
        print(f"  router homophily={self.last_homophily} expert={name}")
        self.active_expert().log_stats()

    def count_parameters(self):
        return self.active_expert().count_parameters()


def _router_kwargs(common):
    skip = {"in_dim", "out_dim", "hidden_dims", "distance", "gamma_mode", "penalty", "gamma", "gamma_bound", "strategy_id", "percentile_q"}
    out = {k: v for k, v in common.items() if k not in skip}
    out["in_dim"] = common["in_dim"]
    out["out_dim"] = common["out_dim"]
    if "hidden_dims" in common:
        out["hidden_dims"] = common["hidden_dims"]
    return out
