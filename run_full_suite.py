#!/usr/bin/env python
"""Pull TAR2003/RUNG_ML_Project, run every algorithm, then draw the charts.

The remote is https://github.com/TAR2003/RUNG_ML_Project.git (branch master).

What "everything" means
    1. git fetch + fast-forward pull. Local edits are stashed and restored.
    2. The unit checks in tests/test_rung_strategy.py.
    3. Clean training, structure PGD, and the cosine adaptive attack for every
       algorithm below, on every dataset below.
    4. python plot_logs.py, which writes figures/attack, figures/cosine_attack,
       and figures/attack_comparison.

Algorithms
    The logged RUNG variants, GCN, GAT, APPNP, MLP, and strategy presets
    S01-S29 and S31-S34. S30 is the same three baselines and is not launched
    as its own model. ogbn-arxiv is left out: the dense pairwise step does
    not fit that graph.

This is a multi-day run. Narrow it while you are checking the wiring:

    python run_full_suite.py --skip-pull --datasets cora --models S07 RUNG --max-epoch 2

Full run:

    python run_full_suite.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REMOTE_URL = "https://github.com/TAR2003/RUNG_ML_Project.git"
BRANCH = "master"

DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "chameleon",
    "cornell",
    "wisconsin",
    "squirrel",
    "actor",
    "texas",
]

# One command per algorithm. Preset ids are accepted by clean.py as --model S07.
ALGORITHMS = [
    "RUNG",
    "RUNG_percentile_gamma",
    "RUNG_learnable_distance",
    "RUNG_parametric_gamma",
    "RUNG_learnable_gamma",
    "RUNG_learnable_combined",
    "RUNG_confidence_lambda",
    "RUNG_homophily_adaptive",
    "RUNG_combined",
    "RUNG_combined_model",
    "GCN",
    "GAT",
    "APPNP",
    "MLP",
    "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09",
    "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19",
    "S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29",
    "S31", "S32", "S33", "S34",
]


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd or ROOT), check=check)


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["git", *args], check=check)


def pull_latest() -> None:
    """Fast-forward master from TAR2003/RUNG_ML_Project. Keep local edits."""
    inside = _git("rev-parse", "--is-inside-work-tree", check=False)
    if inside.returncode != 0:
        raise SystemExit(f"{ROOT} is not a git checkout. Clone {REMOTE_URL} first.")

    remotes = subprocess.run(
        ["git", "remote"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    remote_name = "origin"
    if "origin" not in remotes:
        _git("remote", "add", "origin", REMOTE_URL)
    else:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        if "TAR2003/RUNG_ML_Project" not in url:
            print(f"origin is {url}. Adding remote 'tar2003' -> {REMOTE_URL}")
            if "tar2003" not in remotes:
                _git("remote", "add", "tar2003", REMOTE_URL)
            remote_name = "tar2003"

    _git("fetch", remote_name)

    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    stashed = False
    if dirty:
        print("Working tree has local changes. Stashing them, including untracked files.")
        _git("stash", "push", "--include-untracked", "-m", "run_full_suite")
        stashed = True

    pull = _git("pull", "--ff-only", remote_name, BRANCH, check=False)
    if pull.returncode != 0:
        print(
            "Fast-forward pull failed. The suite will keep going on the current checkout.\n"
            "Resolve the git error above if you need the remote commits first."
        )

    if stashed:
        pop = _git("stash", "pop", check=False)
        if pop.returncode != 0:
            raise SystemExit(
                "git stash pop hit a conflict. Fix the files, then rerun with --skip-pull."
            )


def run_unit_tests() -> bool:
    script = ROOT / "tests" / "test_rung_strategy.py"
    if not script.exists():
        print(f"Missing {script}")
        return False
    result = _run([sys.executable, str(script)], check=False)
    return result.returncode == 0


def run_algorithms(datasets: list[str], models: list[str], max_epoch: int, cosine: bool) -> bool:
    cmd = [
        sys.executable, str(ROOT / "run_all.py"),
        "--datasets", *datasets,
        "--models", *models,
        "--max_epoch", str(max_epoch),
    ]
    if cosine:
        cmd.append("--cosine-attack")
    result = _run(cmd, check=False)
    return result.returncode == 0


def draw_charts() -> bool:
    result = _run(
        [sys.executable, str(ROOT / "plot_logs.py"), "--log_dir", "log", "--out_dir", "figures"],
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull TAR2003/RUNG_ML_Project, run every algorithm, draw charts.")
    parser.add_argument("--skip-pull", action="store_true", help="Use the checkout as it is.")
    parser.add_argument("--unit-only", action="store_true", help="Run tests/test_rung_strategy.py and stop.")
    parser.add_argument("--skip-unit", action="store_true")
    parser.add_argument("--skip-cosine", action="store_true", help="Skip the cosine adaptive attack.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=ALGORITHMS)
    parser.add_argument("--max-epoch", type=int, default=300)
    args = parser.parse_args()

    started = time.perf_counter()
    report = {"unit": None, "algorithms": None, "charts": None}

    if not args.skip_pull:
        pull_latest()

    if not args.skip_unit:
        report["unit"] = run_unit_tests()
        if not report["unit"]:
            print("Unit checks failed. Algorithm runs still start so one failure does not hide the rest.")

    if not args.unit_only:
        print(
            f"\nLaunching {len(args.models)} algorithms x {len(args.datasets)} datasets "
            f"for {args.max_epoch} epochs, with structure attack"
            f"{'' if args.skip_cosine else ' and cosine attack'}."
        )
        report["algorithms"] = run_algorithms(
            args.datasets, args.models, args.max_epoch, cosine=not args.skip_cosine,
        )
        report["charts"] = draw_charts()

    report["seconds"] = round(time.perf_counter() - started, 1)
    out = ROOT / "figures" / "suite_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSuite report:", report)
    print("Charts:", ROOT / "figures")

    if report["unit"] is False or report["algorithms"] is False or report["charts"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
