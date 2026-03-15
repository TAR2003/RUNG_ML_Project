#!/usr/bin/env python
"""
plot_logs.py — Read all log files and generate figures automatically.

Scans log/<dataset>/clean/ and log/<dataset>/attack/ for *every* log written
by run_all.py (clean.py + attack.py), extracts the accuracy numbers, and
produces:

  figures/robustness_<dataset>.png
      One figure per dataset.  X-axis = attack-budget ratio, Y-axis = accuracy.
      Budget 0 is the clean (unattacked) accuracy; subsequent points are the
      attacked accuracy at each budget.  Each model gets its own coloured line
      with ±1 std shading.

  figures/clean_accuracy.png
      Bar chart comparing clean accuracy across all models and datasets.

Usage
-----
    python plot_logs.py
    python plot_logs.py --log_dir log --out_dir figures
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # headless — no GUI needed
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate figures from training/attack logs.")
parser.add_argument("--log_dir", type=str, default="log",
                    help="Root directory that holds per-dataset log sub-folders (default: log/).")
parser.add_argument("--out_dir", type=str, default="figures",
                    help="Directory where PNG figures are saved (default: figures/).")
parser.add_argument("--dpi", type=int, default=150,
                    help="Figure resolution in DPI (default: 150).")
args = parser.parse_args()

LOG_DIR = Path(args.log_dir)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_RE_BUDGET    = re.compile(r"^Budget:\s*([\d.]+)", re.IGNORECASE)
_RE_CLEAN_SUM = re.compile(r"Clean:\s*([\d.]+)[^±]*±([\d.]+)", re.UNICODE)
_RE_ATK_SUM   = re.compile(r"Attacked:\s*([\d.]+)[^±]*±([\d.]+)", re.UNICODE)
# model done line from clean.py: "model RUNG done, clean acc: 0.811±0.008"
_RE_CLEAN_DONE = re.compile(r"model .+ done,\s*clean acc:\s*([\d.]+)[^±]*±([\d.]+)", re.IGNORECASE)
# Per-split pair lines in attack logs: two floats on one line
_RE_PAIR = re.compile(r"^([\d.]+)\s+([\d.]+)\s*$")
# Alternative format: consecutive Clean/Attacked lines without Budget prefix (legacy format)
_RE_CLEAN_LINE = re.compile(r"^\s*Clean:\s*([\d.]+)[^±]*±([\d.]+)", re.UNICODE)
_RE_ATK_LINE   = re.compile(r"^\s*Attacked:\s*([\d.]+)[^±]*±([\d.]+)", re.UNICODE)

# ---------------------------------------------------------------------------
# Label builder
# ---------------------------------------------------------------------------

def _attack_label(stem: str) -> str:
    """
    Convert an attack log filename stem such as
        RUNG_normMCP_gamma6.0
        RUNG_new_SCAD_normSCAD_gamma6.0
    into a short readable label:
        RUNG (MCP, γ=6.0)
        RUNG_new_SCAD (SCAD, γ=6.0)
    """
    m = re.match(r"^(.+?)_norm(.+?)_gamma([\d.]+)$", stem)
    if m:
        model, norm, gamma = m.group(1), m.group(2), m.group(3)
        # drop trailing .0 from gamma for display
        gamma_str = gamma.rstrip("0").rstrip(".")
        return f"{model} ({norm}, γ={gamma_str})"
    return stem   # fallback: use raw stem


def _clean_label(stem: str) -> str:
    """
    Convert a clean log filename stem such as  RUNG_MCP_6.0  into a label.
    Format is   <model>_<norm>_<gamma>   where model may itself contain _.
    """
    # Split off last two tokens (norm and gamma)
    parts = stem.rsplit("_", 2)
    if len(parts) == 3:
        model, norm, gamma = parts
        gamma_str = gamma.rstrip("0").rstrip(".")
        return f"{model} ({norm}, γ={gamma_str})"
    return stem


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_attack_log(path: Path) -> dict:
    """
    Returns a dict:
        {
          budget (float): {
              "clean_mean": float,
              "clean_std":  float,
              "atk_mean":   float,
              "atk_std":    float,
          },
          ...
        }
    Supports both new format (with Budget: sections) and legacy formats.
    Clean accuracy at budget 0 is set from the first "Clean:" summary line found.
    """
    text = path.read_text(errors="replace")
    sections: dict = {}
    current_budget = None
    
    # First pass: try new format with Budget: sections
    for line in text.splitlines():
        line = line.strip()
        # Normalise unicode minus / non-breaking spaces sometimes in output
        line = line.replace("\u00c2", "").replace("\u00b1", "±")

        m = _RE_BUDGET.match(line)
        if m:
            current_budget = float(m.group(1))
            sections[current_budget] = {}
            continue

        if current_budget is None:
            continue

        mc = _RE_CLEAN_SUM.search(line)
        if mc:
            sections[current_budget]["clean_mean"] = float(mc.group(1))
            sections[current_budget]["clean_std"]  = float(mc.group(2))
            continue

        ma = _RE_ATK_SUM.search(line)
        if ma:
            sections[current_budget]["atk_mean"] = float(ma.group(1))
            sections[current_budget]["atk_std"]  = float(ma.group(2))

    # Keep only fully-parsed sections (both clean + attacked present)
    sections = {b: v for b, v in sections.items()
                if "clean_mean" in v and "atk_mean" in v}
    
    # If new format didn't work, try legacy format (consecutive Clean:/Attacked: without Budget prefix)
    if not sections:
        pending_clean = None
        budget_counter = 0.0
        
        for line in text.splitlines():
            line = line.strip()
            line = line.replace("\u00c2", "").replace("\u00b1", "±")
            
            # Try to match legacy format's Clean line
            mc = _RE_CLEAN_LINE.match(line)
            if mc:
                pending_clean = (float(mc.group(1)), float(mc.group(2)))
                continue
            
            # Try to match legacy format's Attacked line
            # Only if we have a pending Clean value
            if pending_clean:
                ma = _RE_ATK_LINE.match(line)
                if ma:
                    clean_mean, clean_std = pending_clean
                    atk_mean, atk_std = float(ma.group(1)), float(ma.group(2))
                    # Assign a synthetic budget value based on order
                    # (legacy format doesn't have explicit budgets)
                    sections[budget_counter] = {
                        "clean_mean": clean_mean,
                        "clean_std": clean_std,
                        "atk_mean": atk_mean,
                        "atk_std": atk_std,
                    }
                    budget_counter += 0.1
                    pending_clean = None
    
    return sections


def parse_clean_log(path: Path) -> tuple[float | None, float | None]:
    """
    Returns (mean_accuracy, std_accuracy) from the log, or (None, None).
    Tries multiple formats:
    1. Standard clean log format: "model … done, clean acc: X±Y" line
    2. Legacy/alternative format: First "Clean: X±Y" line found
    """
    mean_, std_ = None, None
    text = path.read_text(errors="replace")
    
    # Try standard clean log format first
    for line in text.splitlines():
        line = line.strip().replace("\u00c2", "").replace("\u00b1", "±")
        m = _RE_CLEAN_DONE.search(line)
        if m:
            mean_ = float(m.group(1))
            std_  = float(m.group(2))
            return mean_, std_  # Return immediately on first match for consistency
    
    # Fallback: try to find "Clean:" summary line (legacy or alternative format)
    if mean_ is None:
        for line in text.splitlines():
            line = line.strip().replace("\u00c2", "").replace("\u00b1", "±")
            mc = _RE_CLEAN_LINE.match(line)
            if mc:
                mean_ = float(mc.group(1))
                std_  = float(mc.group(2))
                break  # Take first "Clean:" line found
    
    return mean_, std_


def _detect_log_type(path: Path) -> str:
    """
    Detect whether a log file contains attack data, clean data, or both.
    Returns: "attack", "clean", "mixed", or "unknown"
    This allows graceful handling of logs in wrong directories or mixed content.
    """
    text = path.read_text(errors="replace")
    has_budget = bool(re.search(r"^Budget:", text, re.MULTILINE | re.IGNORECASE))
    has_attacked = bool(re.search(r"Attacked:", text))
    has_clean_sum = bool(re.search(r"^\s*Clean:", text, re.MULTILINE))
    has_model_done = bool(re.search(r"model .+ done,\s*clean acc:", text, re.IGNORECASE))
    
    # New format: Budget sections with Attacked summaries = attack data
    if has_budget and has_attacked:
        return "attack"
    
    # Legacy format: Attacked summaries without Budget = attack data
    if has_attacked:
        return "attack"
    
    # Clean format: model done line = clean data
    if has_model_done:
        return "clean"
    
    # Fallback: if none of above, it's unknown
    return "unknown"


# ---------------------------------------------------------------------------
# Discover all logs
# ---------------------------------------------------------------------------

# Structure we build:
#   attack_data[dataset][label] = { budget: {clean_mean,clean_std,atk_mean,atk_std} }
#   clean_data[dataset][label]  = (mean, std)

attack_data: dict[str, dict[str, dict]] = defaultdict(dict)
clean_data:  dict[str, dict[str, tuple]] = defaultdict(dict)

if not LOG_DIR.exists():
    print(f"ERROR: log directory '{LOG_DIR}' does not exist.")
    sys.exit(1)

for dataset_dir in sorted(LOG_DIR.iterdir()):
    if not dataset_dir.is_dir():
        continue
    dataset = dataset_dir.name

    # ── attack logs ────────────────────────────────────────────────────────
    attack_dir = dataset_dir / "attack"
    if attack_dir.is_dir():
        for log_file in sorted(attack_dir.glob("*.log")):
            log_type = _detect_log_type(log_file)
            
            # If it's actually clean data in attack dir, skip it here
            # (it will be picked up in clean_dir processing if it exists there)
            if log_type == "clean":
                print(f"  [attack] {dataset}/{log_file.name}  →  detected as clean format (skipped from attack)")
                continue
            
            if log_type == "attack":
                label   = _attack_label(log_file.stem)
                budgets = parse_attack_log(log_file)
                if budgets:
                    attack_data[dataset][label] = budgets
                    print(f"  [attack] {dataset}/{log_file.name}  →  "
                          f"{len(budgets)} budget(s)  label='{label}'")
                else:
                    print(f"  [attack] {dataset}/{log_file.name}  →  no parseable data (skipped)")
            else:
                print(f"  [attack] {dataset}/{log_file.name}  →  unknown format (skipped)")

    # ── clean logs ─────────────────────────────────────────────────────────
    clean_dir = dataset_dir / "clean"
    if clean_dir.is_dir():
        for log_file in sorted(clean_dir.glob("*.log")):
            log_type = _detect_log_type(log_file)
            
            if log_type == "attack":
                # Attack data in clean dir - use appropriate labeling
                label = _attack_label(log_file.stem)
                budgets = parse_attack_log(log_file)
                if budgets:
                    attack_data[dataset][label] = budgets
                    print(f"  [attack] {dataset}/{log_file.name}  →  "
                          f"{len(budgets)} budget(s)  label='{label}' (found in clean/)")
                else:
                    print(f"  [attack] {dataset}/{log_file.name}  →  no parseable data (skipped)")
                continue
            
            if log_type == "clean":
                label = _clean_label(log_file.stem)
                mean_, std_ = parse_clean_log(log_file)
                if mean_ is not None:
                    clean_data[dataset][label] = (mean_, std_)
                    print(f"  [clean]  {dataset}/{log_file.name}  →  "
                          f"acc={mean_:.4f}±{std_:.4f}  label='{label}'")
                else:
                    print(f"  [clean]  {dataset}/{log_file.name}  →  no valid data (skipped)")
            else:
                print(f"  [clean]  {dataset}/{log_file.name}  →  unknown format (skipped)")

if not attack_data and not clean_data:
    print("\nNo log data found.  Run  python run_all.py  first.")
    sys.exit(0)

print()  # blank line before figures section

# Optional per-dataset comparison model list (one model match per line).
# If present, an extra figure is generated for each dataset showing only
# those models that match the list.
COMPARE_MODELS_PATH = Path("compare_models.txt")
# `compare_models` is populated after the helper functions are defined.
compare_models: list[str] = []

# ---------------------------------------------------------------------------
# Colour palette (cycles automatically for many models)
# ---------------------------------------------------------------------------
PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # teal
]


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _read_compare_list(path: Path) -> list[str]:
    """Read optional list of model name filters for additional comparison plots."""
    if not path.exists():
        return []
    lines = [l.strip() for l in path.read_text(errors="replace").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]


def _filter_labels(labels: list[str], queries: list[str]) -> list[str]:
    """Return subset of labels that match any query (case-insensitive substring)."""
    if not queries:
        return []
    out = []
    for lbl in labels:
        lower = lbl.lower()
        if any(q.lower() in lower for q in queries):
            out.append(lbl)
    return out


# Read optional compare list (one model pattern per line).
# When present, an extra per-dataset plot is generated showing only these models.
compare_models = _read_compare_list(COMPARE_MODELS_PATH)
if compare_models:
    print(f"Using compare list from '{COMPARE_MODELS_PATH}': {compare_models}")

# ---------------------------------------------------------------------------
# Figure 1: per-dataset robustness curves
# ---------------------------------------------------------------------------
# Sort datasets for deterministic output
all_datasets = sorted(set(list(attack_data.keys()) + list(clean_data.keys())))

for dataset in all_datasets:
    a_data = attack_data.get(dataset, {})
    c_data = clean_data.get(dataset, {})

    if not a_data:
        print(f"  [skip] No attack data for dataset '{dataset}' — robustness figure skipped.")
        continue

    # Collect all model labels present in attack data
    # Use attack label → match to clean label heuristically
    all_labels = sorted(a_data.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Robustness under PGD attack — {dataset}", fontsize=13, pad=10)
    ax.set_xlabel("Attack budget (fraction of edges)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    for idx, label in enumerate(all_labels):
        budgets_dict = a_data[label]
        budgets_sorted = sorted(budgets_dict.keys())

        # --- x-axis points ---
        # Budget 0.0 = clean accuracy (take from attack log's "Clean:" summary,
        # which is constant across budgets; fall back to clean log if available)
        first_budget_data = budgets_dict[budgets_sorted[0]]
        clean_mean = first_budget_data["clean_mean"]
        clean_std  = first_budget_data["clean_std"]

        # Prefer clean log if a matching label is found
        if label in c_data:
            cm, cs = c_data[label]
            clean_mean, clean_std = cm, cs

        xs     = [0.0]       + budgets_sorted
        means  = [clean_mean]  + [budgets_dict[b]["atk_mean"] for b in budgets_sorted]
        stds   = [clean_std]   + [budgets_dict[b]["atk_std"]  for b in budgets_sorted]

        color = _color(idx)
        ax.plot(xs, means, marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(
            xs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15,
            color=color,
        )

    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=0.0, top=1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()

    out_path = OUT_DIR / f"robustness_{dataset}.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # --- Optional compare-list figure (per-dataset) ---
    if compare_models:
        filtered_labels = _filter_labels(all_labels, compare_models)
        if filtered_labels:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(f"Robustness (compare list) — {dataset}", fontsize=13, pad=10)
            ax.set_xlabel("Attack budget (fraction of edges)", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)

            for idx, label in enumerate(filtered_labels):
                budgets_dict = a_data[label]
                budgets_sorted = sorted(budgets_dict.keys())

                first_budget_data = budgets_dict[budgets_sorted[0]]
                clean_mean = first_budget_data["clean_mean"]
                clean_std  = first_budget_data["clean_std"]
                if label in c_data:
                    cm, cs = c_data[label]
                    clean_mean, clean_std = cm, cs

                xs     = [0.0]       + budgets_sorted
                means  = [clean_mean]  + [budgets_dict[b]["atk_mean"] for b in budgets_sorted]
                stds   = [clean_std]   + [budgets_dict[b]["atk_std"]  for b in budgets_sorted]

                color = _color(idx)
                ax.plot(xs, means, marker="o", linewidth=2, label=label, color=color)
                ax.fill_between(
                    xs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15,
                    color=color,
                )

            ax.set_xlim(left=-0.01)
            ax.set_ylim(bottom=0.0, top=1.05)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.legend(loc="lower right", fontsize=6, framealpha=0.9)
            fig.tight_layout()

            out_path = OUT_DIR / f"robustness_{dataset}_compare.png"
            fig.savefig(out_path, dpi=args.dpi)
            plt.close(fig)
            print(f"  Saved: {out_path}")
        else:
            print(f"  [compare] {dataset}: no models matched compare list (skipped)")

    # --- Pairwise RUNG vs other model plots ---
    rung_label = None
    for label in all_labels:
        if label.startswith("RUNG"):
            rung_label = label
            break
    if rung_label:
        for other_label in all_labels:
            if other_label == rung_label:
                continue
            # Plot RUNG and other_label together
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(f"Robustness: {dataset} — {rung_label} vs {other_label}", fontsize=13, pad=10)
            ax.set_xlabel("Attack budget (fraction of edges)", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)

            for idx, label in enumerate([rung_label, other_label]):
                budgets_dict = a_data[label]
                budgets_sorted = sorted(budgets_dict.keys())
                first_budget_data = budgets_dict[budgets_sorted[0]]
                clean_mean = first_budget_data["clean_mean"]
                clean_std  = first_budget_data["clean_std"]
                if label in c_data:
                    cm, cs = c_data[label]
                    clean_mean, clean_std = cm, cs
                xs     = [0.0]       + budgets_sorted
                means  = [clean_mean]  + [budgets_dict[b]["atk_mean"] for b in budgets_sorted]
                stds   = [clean_std]   + [budgets_dict[b]["atk_std"]  for b in budgets_sorted]
                color = _color(idx)
                ax.plot(xs, means, marker="o", linewidth=2, label=label, color=color)
                ax.fill_between(
                    xs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15,
                    color=color,
                )
            ax.set_xlim(left=-0.01)
            ax.set_ylim(bottom=0.0, top=1.05)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
            fig.tight_layout()
            safe_other = other_label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "").replace("γ", "gamma")
            safe_rung = rung_label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "").replace("γ", "gamma")
            out_path = OUT_DIR / f"robustness_{dataset}_{safe_rung}_vs_{safe_other}.png"
            fig.savefig(out_path, dpi=args.dpi)
            plt.close(fig)
            print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: clean accuracy bar chart (all datasets × all models)
# ---------------------------------------------------------------------------
# Build a unified set of models across all datasets; collect values
all_model_labels = sorted(
    {lbl for ds_data in clean_data.values() for lbl in ds_data}
    | {lbl for ds_data in attack_data.values() for lbl in ds_data}
)

# For each (model, dataset) give the clean accuracy
#   priority: clean log > attack log's first-budget "clean_mean"
clean_means: dict[str, dict[str, tuple[float, float]]] = {}  # label → dataset → (mean, std)
for label in all_model_labels:
    clean_means[label] = {}
    for dataset in all_datasets:
        if label in clean_data.get(dataset, {}):
            clean_means[label][dataset] = clean_data[dataset][label]
        elif label in attack_data.get(dataset, {}):
            bd = attack_data[dataset][label]
            if bd:
                first = bd[sorted(bd.keys())[0]]
                clean_means[label][dataset] = (first["clean_mean"], first["clean_std"])

# Only draw the bar chart if there is at least one value
has_any = any(clean_means[lbl] for lbl in all_model_labels)
if has_any:
    n_models   = len(all_model_labels)
    n_datasets = len(all_datasets)
    x = np.arange(n_datasets)
    bar_width = min(0.8 / max(n_models, 1), 0.25)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(max(6, n_datasets * 2.5), 5))
    ax.set_title("Clean accuracy by model and dataset", fontsize=13, pad=10)
    ax.set_ylabel("Clean accuracy", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(all_datasets, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.8)

    for i, label in enumerate(all_model_labels):
        ds_vals = clean_means[label]
        bar_means = []
        bar_stds  = []
        for ds in all_datasets:
            if ds in ds_vals:
                bar_means.append(ds_vals[ds][0])
                bar_stds.append(ds_vals[ds][1])
            else:
                bar_means.append(0.0)
                bar_stds.append(0.0)

        bars = ax.bar(
            x + offsets[i],
            bar_means,
            width=bar_width * 0.9,
            yerr=bar_stds,
            capsize=4,
            color=_color(i),
            label=label,
            alpha=0.85,
        )
        # Annotate each bar with the mean value
        for bar, mean_val in zip(bars, bar_means):
            if mean_val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{mean_val:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )

    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()

    out_path = OUT_DIR / "clean_accuracy.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"  Saved: {out_path}")
else:
    print("  [skip] No clean accuracy data found — bar chart skipped.")

# ---------------------------------------------------------------------------
# Figure 3: combined robustness — all datasets in one grid figure
# ---------------------------------------------------------------------------
datasets_with_attack = [ds for ds in all_datasets if attack_data.get(ds)]
if len(datasets_with_attack) > 1:
    ncols = min(len(datasets_with_attack), 3)
    nrows = (len(datasets_with_attack) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 6, nrows * 4.5),
                             squeeze=False)
    fig.suptitle("Robustness curves — all datasets", fontsize=14, y=1.01)

    all_labels_union = sorted(
        {lbl for ds in datasets_with_attack for lbl in attack_data[ds]}
    )

    for ax_idx, dataset in enumerate(datasets_with_attack):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        a_data = attack_data[dataset]
        c_data_ds = clean_data.get(dataset, {})

        ax.set_title(dataset, fontsize=11)
        ax.set_xlabel("Attack budget", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

        for idx, label in enumerate(all_labels_union):
            if label not in a_data:
                continue
            budgets_dict = a_data[label]
            budgets_sorted = sorted(budgets_dict.keys())
            first = budgets_dict[budgets_sorted[0]]
            clean_mean = c_data_ds[label][0] if label in c_data_ds else first["clean_mean"]
            clean_std  = c_data_ds[label][1] if label in c_data_ds else first["clean_std"]
            xs    = [0.0] + budgets_sorted
            means = [clean_mean] + [budgets_dict[b]["atk_mean"] for b in budgets_sorted]
            stds  = [clean_std]  + [budgets_dict[b]["atk_std"]  for b in budgets_sorted]
            color = _color(idx)
            ax.plot(xs, means, marker="o", linewidth=1.8, label=label, color=color)
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.12, color=color)

        ax.set_xlim(left=-0.01)
        ax.set_ylim(0.0, 1.05)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(loc="lower right", fontsize=6, framealpha=0.9)

    # hide unused axes
    for ax_idx in range(len(datasets_with_attack), nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "robustness_all_datasets.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print(f"\nAll figures written to '{OUT_DIR}/'.")
print("To re-run after new training/attack logs appear, just run:")
print("    python plot_logs.py")
