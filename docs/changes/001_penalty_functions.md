# 001 — Alternative Penalty Functions

## Date
2026-02-27

## Motivation
The RUNG paper (Section 3.1) shows MCP reduces estimation bias versus l1
because dρ/dy² = 0 for y >= gamma. SCAD has the same zero-bias region but
with a smoother linear transition. This change enables systematic comparison
of penalty shapes on robustness.

## Mathematical Background

For IRLS edge reweighting, we need dρ(y)/dy² at current edge differences y:

| Penalty | Formula for dρ/dy²                              | Bias   | Zero Region     |
|---------|-------------------------------------------------|--------|-----------------|
| l2      | 1                                               | High   | None (= APPNP)  |
| l1      | 1/(2y)                                          | Medium | None            |
| MCP     | max(0, 1/(2y) − 1/(2γ))                         | Low    | y ≥ γ           |
| SCAD    | 1/(2y) if y<λ; (aλ−y)/((a−1)λ·2y) if λ≤y<aλ; 0 | Low  | y ≥ a·λ         |

Standard SCAD parameter: a = 3.7 (Fan & Li, 2001).

## Convention Note
In this codebase, the `w_func(y)` callable computes:
```
W_ij = d_{y²} ρ(y)  evaluated at  y = ||f_i/√d_i - f_j/√d_j||_2
```
This equals `max(0, 1/2y - 1/2γ)` for MCP — NOT `max(0, 1/y - 1/γ)`.
All new penalties follow the same convention (factor of 2 in denominator).

## What Changed

### New Files
- **`model/penalty.py`** (new): `PenaltyFunction` class with static methods:
  - `PenaltyFunction.mcp(y, gamma)` — lines 51–75
  - `PenaltyFunction.scad(y, lam, a=3.7)` — lines 77–115
  - `PenaltyFunction.l1(y)` — lines 117–131
  - `PenaltyFunction.l2(y)` — lines 133–148
  - `PenaltyFunction.homophily_adaptive(y, gamma, soft_labels)` — lines 150–197
  - `PenaltyFunction.get_w_func(penalty, gamma, a)` — lines 199–228 (factory)

### Modified Files
- **`model/rung.py`**: 
  - Added `from model.penalty import PenaltyFunction` import
  - Added `penalty: str = None` and `gamma: float = 3.0` kwargs to `__init__`
  - Extended `forward()` to branch on `self.penalty == 'adaptive'`
  - Added `get_aggregated_features(A, X)` method
- **`exp/config/get_model.py`**:
  - Added `from model.penalty import PenaltyFunction` import
  - Extended `get_model_default()` to handle `norm ∈ {MCP, SCAD, L1, L2, ADAPTIVE}`
  - SCAD: uses `lam = gamma/3.7` so the cutoff `a*lam = gamma` matches MCP's `gamma`
- **`clean.py`**:
  - Added `--penalty` argument (choices: mcp, scad, l1, l2, adaptive)
  - `--penalty` overrides `--norm` for convenience (lower-case aliases)
- **`attack.py`**:
  - Same `--penalty` argument added

## How to Use

```bash
# Original MCP (unchanged default)
python clean.py --model='RUNG' --norm='MCP' --gamma=6.0 --data='cora'

# Using --penalty alias (lower-case convenience)
python clean.py --model='RUNG' --penalty=mcp  --gamma=6.0 --data='cora'

# SCAD penalty
python clean.py --model='RUNG' --penalty=scad --gamma=6.0 --data='cora'

# L1 penalty (should reproduce RUNG-l1 from paper Table 1)
python clean.py --model='RUNG' --penalty=l1 --data='cora'

# L2 penalty (should approach APPNP / GCN)
python clean.py --model='RUNG' --penalty=l2 --data='cora'

# Adaptive penalty (for heterophilic graphs)
python clean.py --model='RUNG' --penalty=adaptive --gamma=3.0 --data='chameleon'
```

## SCAD Parameter Mapping
When `--penalty=scad --gamma=6.0` is passed, the model sets `lam = 6.0/3.7 ≈ 1.62`
so that the SCAD zero-region threshold `a*lam = 3.7 * 1.62 = 6.0` matches the MCP
threshold for fair comparison.

## Verification
- `--penalty l2` should reproduce APPNP numbers (W_ij = 1 everywhere).
- `--penalty l1` should reproduce RUNG-l1 from paper Table 1.
- `--penalty mcp` should reproduce original RUNG from paper Table 1.
- SCAD should be between l1 and MCP (less bias than l1, comparable to MCP).
