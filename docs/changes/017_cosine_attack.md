# 017 - Adaptive Cosine Distance Attack

## Date

2026-03-19

## What This Is

A new PGD attack variant that targets cosine-distance-based defenses by jointly optimizing:

- Misclassification objective (same core margin objective as existing PGD flow)
- Stealth objective on inserted edges via cosine distance

Cosine adaptive objective used in this update:

- `L = L_margin + beta * mean_cosine_distance(added_edges)`

The optimizer minimizes this objective in the existing projected gradient descent routine, so a lower margin and lower cosine distance are both encouraged.

## Files Created

- `cosine_attack.py` - attack entry point parallel to `attack.py`, writes to `log/{dataset}/cosine_attack/`
- `docs/changes/017_cosine_attack.md` - this change note

## Files Modified

- `train_test_combined.py` - added `attack_pgd_cosine()` and optional cosine attack logging path for `RUNG_combined`
- `run_all.py` - added Step 3 orchestration for cosine attack and new CLI controls

## Key Arguments

- `--cosine_attack_epochs 100` - PGD iterations for Step 3
- `--beta 1.0` - stealth weight for cosine edge-distance term
- `--skip_cosine_attack` - skip Step 3 and keep legacy behavior

## How To Run

```bash
python run_all.py --models RUNG RUNG_combined \
                  --datasets cora citeseer \
                  --cosine_attack_epochs 100 \
                  --beta 1.0
```

## Step Routing Details

- Non-`RUNG_combined` models:
  - Step 1 via `clean.py`
  - Step 2 via `attack.py`
  - Step 3 via `cosine_attack.py`
- `RUNG_combined`:
  - Uses existing unified `train_test_combined.py` flow
  - Step 3 is integrated in that unified run when not skipped

## Expected Outcomes

If `RUNG_combined` remains strong under cosine adaptive attack:

- This supports a stronger robustness claim against defense-aware attackers.

If `RUNG_combined` drops under cosine adaptive attack:

- This exposes an adaptive weakness and informs next defense iterations.

## Beta Sweep Note

Recommended ablation: `beta in {0.5, 1.0, 2.0}`.
A `beta=0.0` run removes stealth pressure and reduces to a stronger pure-PGD variant under the same iteration budget.
