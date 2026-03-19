# 018 - RUNG_homophily_adaptive

## Date
2026-03-19

## Motivation (from progress report)
"Combined approach is highly effective for homophilic graphs,
heterophilic graphs suffer from over-regularization when dynamic
thresholding is combined with dynamic metrics."

This is the double-correction problem: aggressive cosine+percentile
pruning removes valid cross-class edges on heterophilic graphs.

## Solution
Per-node adaptive percentile q based on soft local homophily:
    h_i = mean prediction similarity to neighbors
    q_i = q_base + (1 - h_i) * q_relax

Heterophilic nodes (low h_i) get higher q_i -> less pruning -> valid edges kept.
Homophilic nodes (high h_i) get lower q_i -> more pruning -> adversarial edges removed.

## Files Created
- model/rung_homophily_adaptive.py
- train_eval_data/fit_homophily_adaptive.py
- test_homophily_adaptive.py
- docs/changes/018_homophily_adaptive.md

## Files Modified
- exp/config/get_model.py
- clean.py
- attack.py
- run_all.py
- utils.py

## Key Arguments
- --q_relax 0.20   main tuning knob
- --q_max 0.99     upper bound on q_i
- --homophily_mode from_F0 or per_layer

## Expected Results
On homophilic (Cora, Citeseer):
    Similar to RUNG_combined (q_relax has little effect when h_i is high)

On heterophilic (Chameleon, Cornell, Wisconsin):
    Better than RUNG_combined (heterophilic nodes get gentler pruning)
    This directly addresses the double-correction problem from the paper

## Ablation Questions
1. What is the optimal q_relax? (search 0.0 to 0.40)
2. Does homophily_mode matter? (from_F0 vs per_layer)
3. Does q_relax=0 reproduce RUNG_combined? (should be approximately yes)
4. What does the h_i distribution look like on hetero vs homo graphs?
   (analyze model._last_h_mean after forward pass)
