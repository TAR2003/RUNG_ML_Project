"""Shared CLI helpers for S01-S34."""

import re


def add_strategy_args(parser):
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Strategy id S01-S34. Also accepted as --model S07.",
    )
    parser.add_argument("--lam_hat", type=float, default=None, help="Override preset skip strength.")
    parser.add_argument("--prop_step", type=int, default=None, help="Override preset propagation steps.")
    parser.add_argument(
        "--train_mode", type=str, default=None, choices=["clean", "consistency", "adversarial"],
        help="Override the preset training loss.",
    )
    parser.add_argument("--consistency_weight", type=float, default=None)
    parser.add_argument("--report_bias", action="store_true")
    parser.add_argument("--homophily_threshold", type=float, default=None)


def normalize_strategy_model(args):
    """Map --model S07 onto RUNG_strategy + --strategy S07."""
    name = getattr(args, "model", "") or ""
    if re.fullmatch(r"S\d+", name):
        args.strategy = name
        args.model = "RUNG_strategy"
    elif getattr(args, "strategy", None):
        args.model = "RUNG_strategy"
    return args


def strategy_config_from_args(args):
    cfg = {
        "strategy": args.strategy,
        "gamma": getattr(args, "gamma", None),
        "percentile_q": getattr(args, "percentile_q", None),
        "decay_rate_init": getattr(args, "decay_rate_init", None),
        "q_relax": getattr(args, "q_relax", None),
        "q_max": getattr(args, "q_max", None),
        "homophily_mode": getattr(args, "homophily_mode", None),
        "lam_hat": getattr(args, "lam_hat", None),
        "prop_step": getattr(args, "prop_step", None),
        "train_mode": getattr(args, "train_mode", None),
        "consistency_weight": getattr(args, "consistency_weight", None),
        "homophily_threshold": getattr(args, "homophily_threshold", None),
    }
    if getattr(args, "report_bias", False):
        cfg["report_bias"] = True
    return cfg
