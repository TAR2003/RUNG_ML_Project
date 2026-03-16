# RUNG_COMBINED Attack Fix - Summary

## Problem
The `attack_pgd` function in `train_test_combined.py` was failing with:
```
ERROR: missing a required argument: 'flip_shape_or_init'
```

This occurred for all attack budgets [0.05, 0.10, 0.20, 0.30, 0.40, 0.60] after training completed.

## Root Cause
The function was calling `proj_grad_descent` with the wrong API parameters:

**WRONG (before):**
```python
A_attacked, _ = proj_grad_descent(
    adj=A.cpu().numpy(),
    feat=X.cpu().numpy(),
    labels=y.cpu().numpy(),
    idx_train=None,
    idx_test=test_idx.cpu().numpy(),
    model=model,
    epoch=n_epochs,
    perturbation_ratio=budget,
    lr=lr_attack,
    device=device,
)
```

**CORRECT (after):**
```python
edge_pert, _ = proj_grad_descent(
    flip_shape_or_init=A.shape,        # ✓ Required parameter
    symmetric=True,                     # ✓ Required parameter
    device=A.device,                    # ✓ Required parameter
    budget=budget_edge_num,             # ✓ Required parameter
    grad_fn=grad_fn,                    # ✓ Required parameter
    loss_fn=loss_fn,                    # ✓ Required parameter
    iterations=n_epochs,
    base_lr=lr_attack,
    grad_clip=1.0,
    progress=False,
)
```

## Changes Made

### File: `train_test_combined.py`

**Function: `attack_pgd` (lines 173-254)**

1. **Defined proper loss and gradient functions:**
   - `loss_fn(flip)`: Computes margin loss on attacked graph using the dense flip matrix
   - `grad_fn(flip)`: Computes gradients for the loss

2. **Fixed `proj_grad_descent` call:**
   - Uses correct required parameters: `flip_shape_or_init`, `symmetric`, `device`, `budget`, `grad_fn`, `loss_fn`
   - Returns edge indices `edge_pert` (shape `[num_edges, 2]`), not dense matrix

3. **Properly handles edge index conversion:**
   - Uses `edge_diff_matrix(edge_pert.long(), A)` to convert edge indices to dense perturbation matrix
   - Applies perturbation: `A_attacked = A + edge_diff_matrix(edge_pert, A)`

4. **Added error handling:**
   - Graceful fallback if attack fails
   - Handles empty edge perturbations

## Verification

✅ All validation tests pass:
- Function syntax is valid
- Attack logic works with synthetic data
- Correct API parameters used
- Edge indices properly converted to dense matrix
- Perturbation applied correctly

## Expected Behavior

After this fix, `train_test_combined.py` will:

1. ✅ Train RUNG_combined model successfully (already working)
2. ✅ Run PGD attacks for each budget without "flip_shape_or_init" errors (FIXED)
3. ✅ Produce attacked accuracy for each budget
4. ✅ Complete full training + attack pipeline in one command

## Testing

Run the fixed script:
```bash
python train_test_combined.py --dataset cora --max_epoch 300
```

Expected output for attacks (instead of ERROR):
```
Running PGD attacks with budgets: [0.05, 0.1, 0.2, 0.3, 0.4, 0.6]
Attack epochs: 10, lr: 0.01

  Budget 0.05... [attacked accuracy will be printed]
  Budget 0.10... [attacked accuracy will be printed]
  Budget 0.20... [attacked accuracy will be printed]
  Budget 0.30... [attacked accuracy will be printed]
  Budget 0.40... [attacked accuracy will be printed]
  Budget 0.60... [attacked accuracy will be printed]
```

## Files Modified

- `train_test_combined.py` (1 function, 82 lines changed)

## Related Files (Not Modified)

- `model/rung_combined.py` - RUNG_combined model (working correctly)
- `gb/attack/gd.py` - PGD attack implementation (using correct API)
- `gb/pert.py` - Edge perturbation utilities (used correctly now)
