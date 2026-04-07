# RUNG Model Codeflow Documentation

This folder contains source-verified codeflow documentation for the original RUNG and all major `rung_*` variants implemented in this repository.

## Files

1. `00_RUNG_base_codeflow.md`
2. `01_RUNG_learnable_gamma_diff_from_RUNG.md`
3. `02_RUNG_parametric_gamma_diff_from_RUNG.md`
4. `03_RUNG_percentile_gamma_diff_from_RUNG.md`
5. `04_RUNG_learnable_distance_diff_from_RUNG.md`
6. `05_RUNG_learnable_combined_diff_from_RUNG.md`
7. `06_RUNG_combined_diff_from_RUNG.md`
8. `07_RUNG_homophily_adaptive_diff_from_RUNG.md`
9. `08_RUNG_confidence_lambda_diff_from_RUNG.md`
10. `09_RUNG_combined_model_diff_from_RUNG.md`
11. `10_RUNG_new_and_penalty_aliases_diff_from_RUNG.md`
12. `11_RUNG_heterophilic_graph_handling_changes.md`

## Scope Notes

- These docs compare against original `model/rung.py` behavior and default training in `train_eval_data/fit.py`.
- Each file includes concrete code snippets and file references.
- The focus is implementation truth in this repository, not paper-only descriptions.
