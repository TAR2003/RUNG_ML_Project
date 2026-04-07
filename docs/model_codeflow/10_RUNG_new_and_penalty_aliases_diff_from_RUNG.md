# RUNG_new and Penalty Aliases: Differences from Original RUNG

This document covers tweaked variants that are routed through naming/penalty configuration rather than separate `rung_*.py` architecture files.

## Compared Files

- Baseline/Factory: `exp/config/get_model.py`
- Entry dispatch: `clean.py`
- Weight implementations: `model/penalty.py`, `model/att_func.py`
- Core model body remains: `model/rung.py`

---

## 1) What Is Not a Separate Architecture Here

The following names are not separate model classes in `model/`:
- `RUNG_new`
- `RUNG_new_SCAD`
- `RUNG_new_L1`
- `RUNG_new_L2`
- `RUNG_new_ADAPTIVE`
- `RUNG_SCAD`
- `RUNG_L1`
- `RUNG_L2`

They ultimately still instantiate `RUNG(...)` from `model/rung.py`.

---

## 2) Real Difference: Which Weight Function Path Is Used

### Baseline RUNG path in factory

```python
# exp/config/get_model.py
if model_name == 'RUNG':
    ...
    return _build_rung(get_mcp_att_func(gamma)), custom_fit_params
```

This uses `model/att_func.py` constructors such as `get_mcp_att_func`.

### RUNG_new path in factory

```python
# exp/config/get_model.py
elif model_name == 'RUNG_new':
    w_func = PenaltyFunction.get_w_func('mcp', gamma)
    return _build_rung(w_func), custom_fit_params
```

This uses `model/penalty.py` through `PenaltyFunction.get_w_func(...)`.

So the architecture is the same, but the penalty-function implementation source differs.

---

## 3) Compound Alias Normalization in `clean.py`

`clean.py` rewrites compound names into model+norm before model creation:

```python
# clean.py
_COMPOUND_MODEL_MAP = {
    'RUNG_new_SCAD': ('RUNG_new', 'SCAD'),
    'RUNG_new_L1': ('RUNG_new', 'L1'),
    'RUNG_new_L2': ('RUNG_new', 'L2'),
    'RUNG_new_ADAPTIVE': ('RUNG_new', 'ADAPTIVE'),
    'RUNG_SCAD': ('RUNG', 'SCAD'),
    'RUNG_L1': ('RUNG', 'L1'),
    'RUNG_L2': ('RUNG', 'L2'),
}
```

This is a naming/config convenience layer, not a new graph propagation architecture.

---

## 4) Training Path

All these aliases still go through the generic baseline fit branch:

```python
# clean.py
elif args.model in ('RUNG', 'RUNG_new', 'MLP', 'L1', 'APPNP'):
    fit(cur_model, A, X, y, train_idx, val_idx, **train_param)
```

So they do not introduce specialized optimizer groups like the advanced variants.

---

## 5) Honest Full Delta vs Original RUNG

1. No new class-level architecture in `model/`.
2. Main difference is penalty implementation route (`att_func` path vs `PenaltyFunction` path) and norm selection.
3. Same base RUNG forward loop and same default fit routine.
4. Compound names are mostly CLI aliases translated to `(model, norm)`.
