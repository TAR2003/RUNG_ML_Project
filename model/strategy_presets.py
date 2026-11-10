"""Presets S01-S34 from docs/strategy/RUNG_next_strategies.md.

CLI defaults that match the historical RUNG flags (gamma 6, q 0.75,
decay 0.85, q_relax 0.20) do not overwrite a preset. Any other value does.
``lam_hat``, ``prop_step``, and ``train_mode`` overwrite only when the caller
sets them (they have no historical default in the strategy config).
"""

from model.rung_strategy import RUNGRouter, RUNGStrategy


# Keys whose CLI default must not clobber a preset that chose something else.
CLI_DEFAULTS = {
    "gamma": 6.0,
    "percentile_q": 0.75,
    "decay_rate_init": 0.85,
    "q_relax": 0.20,
}

_META = ("train_mode", "report_bias", "consistency_weight", "router", "notes", "kind")


def _p(**kwargs):
    return kwargs


PRESETS = {
    # Family A. Disentangle cosine, percentile, and parametric decay.
    "S01": _p(distance="cosine", gamma_mode="fixed", penalty="mcp", gamma=1.0, notes="cosine + fixed MCP"),
    "S02": _p(distance="cosine", gamma_mode="fixed", penalty="scad", gamma=1.0, notes="cosine + fixed SCAD"),
    "S03": _p(distance="euclidean", gamma_mode="percentile", penalty="scad", notes="Euclidean percentile q sweep"),
    "S04": _p(distance="cosine", gamma_mode="percentile", penalty="scad", notes="cosine percentile q sweep"),
    "S05": _p(distance="cosine", gamma_mode="percentile", penalty="scad", lam_hat=0.7, notes="lam_hat sweep on cosine+q"),
    "S06": _p(distance="cosine", gamma_mode="percentile", penalty="scad", notes="duplicate of RUNG_combined"),
    # Family B. Cosine plus parametric decay, percentile off.
    "S07": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, decay_rate_init=0.85, notes="cosine + parametric"),
    "S08": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, decay_rate_init=0.95, notes="S07 with slower decay"),
    "S09": _p(distance="cosine", gamma_mode="banded", penalty="scad", gamma=1.0, gamma_bound=2.0, notes="parametric prior clamped to a quantile band"),
    # Family C. Heterophily-aware weights.
    "S10": _p(distance="cosine", gamma_mode="per_node_q", penalty="scad", q_relax=0.20, notes="per-node quantile"),
    "S11": _p(distance="euclidean", gamma_mode="fixed", penalty="adaptive", gamma=3.0, notes="inverted MCP"),
    "S12": _p(distance="cosine", gamma_mode="fixed", penalty="adaptive", gamma=1.0, notes="inverted MCP + cosine"),
    "S13": _p(distance="euclidean", gamma_mode="parametric", penalty="adaptive", gamma=3.0, notes="inverted MCP + parametric Euclidean"),
    "S14": _p(distance="cosine", gamma_mode="per_node_q", penalty="scad", q_relax=0.20, notes="per-node q on cosine"),
    "S15": _p(distance="cosine", gamma_mode="percentile", penalty="same_class", cross_class_mode="keep", notes="MCP on same-class edges, keep the rest"),
    "S16": _p(distance="cosine", gamma_mode="percentile", penalty="two_channel", two_channel_beta=0.5, notes="low-pass plus high-pass"),
    # Family D. Gamma inside a live transition zone.
    "S17": _p(distance="cosine", gamma_mode="bounded_schedule", penalty="scad", gamma=1.0, gamma_bound=2.0, notes="bounded decay schedule"),
    "S18": _p(distance="cosine", gamma_mode="bounded_per_layer", penalty="scad", gamma_bound=2.0, notes="K bounded gammas"),
    "S19": _p(distance="cosine", gamma_mode="per_node", penalty="scad", gamma=1.0, gamma_bound=2.0, notes="per-node parametric gamma"),
    # Family E. Skip, readout, representation.
    "S20": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, lambda_mode="confidence", notes="confidence lambda on S07"),
    "S21": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, lambda_mode="layer", lambda_schedule=1.2, notes="layer-wise skip"),
    "S22": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, readout="jump", notes="jumping knowledge"),
    "S23": _p(distance="hidden_cosine", gamma_mode="percentile", penalty="scad", notes="64-d cosine, class-space aggregation"),
    "S24": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, early_exit=True, notes="early exit"),
    "S25": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, detach_warmup_epochs=10, notes="un-detached warmup"),
    # Family F. Training objective. Architecture is S07.
    "S26": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, train_mode="adversarial", notes="curriculum PGD"),
    "S27": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, train_mode="consistency", consistency_weight=0.1, notes="clean/perturbed consistency"),
    # Family G. S28 is the cosine+percentile defense to re-attack. S29 turns bias logging on. S30 is the existing baselines.
    "S28": _p(distance="cosine", gamma_mode="percentile", penalty="scad", notes="defense evaluated by cosine_attack.py"),
    "S29": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, report_bias=True, notes="log estimation bias"),
    "S30": _p(kind="baselines", notes="GCN, GAT, APPNP"),
    # Family H. Same estimator, lower cost.
    "S31": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, edge_only=True, notes="edge-only distance and weights"),
    "S32": _p(distance="cosine", gamma_mode="percentile", penalty="scad", prop_step=5, notes="shorter propagation"),
    "S33": _p(distance="cosine", gamma_mode="parametric", penalty="scad", gamma=1.0, gamma_bound=2.0, fp16_distance=True, notes="float16 distance"),
    # Family I.
    "S34": _p(router=True, homophily_threshold=0.5, notes="homophily router"),
}


SWEEPS = {
    "S01": [{"strategy": "S01", "gamma": g} for g in (0.3, 0.5, 0.8, 1.0, 1.5)],
    "S02": [{"strategy": "S02", "gamma": g} for g in (0.5, 1.0, 1.5)],
    "S03": [{"strategy": "S03", "percentile_q": q} for q in (0.50, 0.75, 0.90, 0.95, 0.99)],
    "S04": [{"strategy": "S04", "percentile_q": q} for q in (0.50, 0.75, 0.90, 0.95, 0.99)],
    "S05": [{"strategy": "S05", "lam_hat": v} for v in (0.5, 0.7, 0.9, 0.95)],
    "S08": [{"strategy": "S07", "decay_rate_init": r} for r in (0.70, 0.85, 0.95)],
    "S10": [{"strategy": "S10", "q_relax": r} for r in (0.0, 0.10, 0.20, 0.40)],
    "S11": [{"strategy": "S11", "gamma": g} for g in (1.0, 3.0, 6.0)],
    "S32": [{"strategy": "S04", "prop_step": k} for k in (2, 5, 10)],
}

BASELINES = ("GCN", "GAT", "APPNP")


def resolve_preset(strategy_id, cfg=None):
    """Return the preset dict with CLI-style overrides applied."""
    if strategy_id not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown strategy '{strategy_id}'. Known: {known}")
    merged = dict(PRESETS[strategy_id])
    cfg = dict(cfg or {})
    for key, default in CLI_DEFAULTS.items():
        if key in cfg and cfg[key] is not None and cfg[key] != default:
            merged[key] = cfg[key]
    for key in ("lam_hat", "prop_step", "train_mode", "consistency_weight", "homophily_threshold", "q_max", "homophily_mode"):
        if cfg.get(key) is not None and key in cfg:
            merged[key] = cfg[key]
    merged["strategy_id"] = strategy_id
    return merged


def build_strategy_model(in_dim, out_dim, cfg, hidden_dims=None):
    """Build the module described by a resolved preset."""
    if cfg.get("kind") == "baselines":
        raise ValueError(
            "S30 is the existing baselines. Run --model GCN, --model GAT, and --model APPNP."
        )
    sid = cfg.get("strategy_id", "custom")
    train_mode = cfg.get("train_mode", "clean")
    report_bias = bool(cfg.get("report_bias", False))
    consistency_weight = float(cfg.get("consistency_weight", 0.1))
    init = {k: v for k, v in cfg.items() if k not in _META and k != "strategy_id" and k != "kind"}
    if hidden_dims is not None:
        init["hidden_dims"] = hidden_dims
    if cfg.get("router"):
        thr = init.pop("homophily_threshold", 0.5)
        model = RUNGRouter(in_dim, out_dim, homophily_threshold=thr, **init)
    else:
        model = RUNGStrategy(in_dim, out_dim, strategy_id=sid, **init)
    model.train_mode = train_mode
    model.report_bias = report_bias
    model.consistency_weight = consistency_weight
    model.strategy_id = sid
    return model
