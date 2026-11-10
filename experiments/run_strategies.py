"""List and launch the S01-S34 presets.

    python experiments/run_strategies.py --list
    python experiments/run_strategies.py --show S07
    python experiments/run_strategies.py --launch S07 --dataset cora --max-epoch 2
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.strategy_presets import BASELINES, PRESETS, SWEEPS


def main():
    parser = argparse.ArgumentParser(description="List or launch a RUNG strategy preset.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show", type=str, default=None, help="Print one preset, for example S07.")
    parser.add_argument("--launch", type=str, default=None, help="Train one preset via clean.py.")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--extra", nargs="*", default=[], help="Extra args passed to clean.py.")
    args = parser.parse_args()

    if args.list or (args.show is None and args.launch is None):
        for sid in sorted(PRESETS):
            notes = PRESETS[sid].get("notes", "")
            print(f"{sid:4}  {notes}")
        print("Sweeps:", ", ".join(sorted(SWEEPS)))
        print("Baselines (S30):", ", ".join(BASELINES))
        if args.launch is None and args.show is None:
            return

    if args.show:
        print(args.show, PRESETS[args.show])
        if args.show in SWEEPS:
            print("sweep:")
            for row in SWEEPS[args.show]:
                print(" ", row)

    if args.launch:
        sid = args.launch
        if sid == "S30":
            print("S30 launches the existing baselines:", ", ".join(BASELINES))
            return
        if sid not in PRESETS:
            raise SystemExit(f"Unknown strategy {sid}")
        cmd = [
            sys.executable, "clean.py",
            "--model", sid,
            "--data", args.dataset,
            "--max_epoch", str(args.max_epoch),
            *args.extra,
        ]
        print(" ".join(cmd))
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
