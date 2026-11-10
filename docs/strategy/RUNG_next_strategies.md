# RUNG Next Strategies

This document is the roadmap for making RUNG more accurate, more robust, and cheaper to run. It records the algorithm as implemented in this repository, the clean and attack numbers already in `log/`, and the strategies worth running next. It does not change model code.

Numbers are means over the five deterministic splits already used by `clean.py` and `attack.py`. Attack numbers are structure PGD at the stated edge-perturbation budget. Source files are named under each table.

## 1. Verdict from the logs

Three facts should drive every next experiment.

1. **Directional distance is the largest robustness gain.** Replacing degree-normalized Euclidean gaps with cosine distance, while keeping percentile gamma at `q = 0.75`, holds accuracy almost flat as the attack budget goes from 5% to 100% on Cora, Pubmed, Chameleon, and Cornell. Base MCP with `γ = 6` loses 15 to 28 points over that same range.
2. **One stack does not win every graph.** `RUNG_combined_model` (cosine + a learnable blend of parametric decay and percentile gamma) is the best model on Cora. Parametric decay alone is the best model on Citeseer and on Wisconsin. Cosine plus percentile is the best model on Pubmed, Cornell, and Chameleon. Blending all three mechanisms drops Wisconsin clean accuracy from 70.4% to 47.4%.
3. **Several implemented ideas have no measured result.** Confidence-weighted lambda, the homophily-adaptive percentile, the inverted MCP penalty, bounded learnable gamma on cosine distance, projection and bilinear distance, and cosine with a fixed gamma are in the code and absent from `log/`. Curriculum adversarial training was run and all 7 runs failed, so it is unmeasured.

The missing cell with the highest expected value is **cosine distance plus parametric decay, with percentile gamma turned off**. Those two wins have never been measured together without the blend that damaged WebKB.

## 2. Current algorithm

Base class: `model/rung.py`. Training loop: `train_eval_data/fit.py`. Factory defaults: `exp/config/get_model.py`.

RUNG (NeurIPS 2024) is a decoupled smoother. An MLP maps raw features into class space. Ten quasi-Newton IRLS steps then robustly smooth those logits on the graph. The final logits are the classifier. There is no separate head on top of `F^(K)`.

### 2.1 Forward pass

```text
F0 = MLP(X)                         # hidden width 64, dropout 0.5, output dim = classes
A  = A + I
d  = A 1
Ã  = D^{-1/2} A D^{-1/2}
λ  = 1 / lam_hat - 1                # logged runs use lam_hat = 0.9, so λ ≈ 0.111
F  = F0

for k in 1..K:                      # K = 10
    y_ij = || f_i/√d_i - f_j/√d_j ||_2     # all pairs, then detach
    W_ij = w(y_ij)                         # MCP in the logged baseline
    W_ii = 0
    Q̂_i = (Σ_j W_ij A_ij) / d_i + λ
    F_i  = (Σ_j W_ij Ã_ij F_j + λ F0_i) / Q̂_i
return F
```

The objective being minimized is

```text
H(F) = Σ_{(i,j) in E} ρ_γ( ||f_i/√d_i - f_j/√d_j|| ) + λ Σ_i ||f_i - f_i^0||^2
```

Logged baseline penalty is MCP, from `model/penalty.py`:

```text
W_ij = max( 0,  1/(2 y_ij) - 1/(2 γ) )
```

with `γ = 6`. Small gaps keep a large weight. Gaps at or above `γ` are zeroed, which is the unbiased-pruning property. SCAD, L1, and L2 are the other closed-form choices. L2 makes every weight 1 and reduces the update to APPNP.

### 2.2 What the implementation actually computes

- `y` is detached. The penalty weights do not send gradient into the features. Gradient reaches the MLP through the skip term `λ F0 / Q̂` and through the linear smoothing of `F`.
- `y` and `W` are dense `N × N`, then multiplied by `A`. Only edges change `F`. Non-edges are computed and thrown away. Cost is `O(K N^2 C)` with `C` equal to the number of classes. That is acceptable on Cora and painful on Pubmed, and it does not fit ogbn-arxiv (about 169k nodes) as a dense matrix.
- Self-loops are added before the degree is computed. The diagonal of `W` is then forced to 0, so a node is not penalized for differing from itself. The self-loop still changes the degree and `Ã`.
- Training is Adam plus cross-entropy on the training nodes, for up to 300 epochs, with early stopping on validation accuracy. There is no robust loss in the base loop.

### 2.3 Knobs that already exist

| Knob | Where it lives | Logged setting |
| --- | --- | --- |
| Penalty `ρ` | `model/penalty.py`, `model/att_func.py` | MCP, `γ = 6` |
| Gamma schedule | percentile, parametric, per-layer, blend | see lineage below |
| Distance inside `y` | `model/rung_learnable_distance.py` | cosine on the logged advanced runs |
| Skip `λ` | scalar in base; per-node in `model/rung_confidence_lambda.py` | scalar `lam_hat = 0.9` |
| Steps `K` | `prop_step` | 10 |
| Training attack | `train_eval_data/adversarial_trainer.py` | implemented, 7/7 runs failed |

## 3. Variant lineage

```text
RUNG (Euclidean, fixed MCP γ)
├── RUNG_new_SCAD / L1 / L2 / ADAPTIVE
│     └── RUNG_learnable_gamma          K free SCAD thresholds, Euclidean
│           ├── RUNG_parametric_gamma   γ^(k) = γ0 * r^k          (2 scalars)
│           ├── RUNG_percentile_gamma   γ^(k) = quantile(y_E, q)  (0 scalars)
│           └── RUNG_confidence_lambda  per-node λ_i from softmax(F0)
├── RUNG_learnable_distance             percentile γ + cosine | projection | bilinear
│     ├── RUNG_combined                 percentile γ + cosine, 0 new scalars
│     └── RUNG_learnable_combined       cosine + γ bounded in (0, 2)
├── RUNG_combined_model                 cosine + blend(parametric, percentile)
├── RUNG_homophily_adaptive             cosine-style stack + per-node q_i(h_i)
└── RUNG_percentile_adv / RUNG_parametric_adv
      same architecture, curriculum PGD loss
```

Naming trap, because it changes how the tables must be read:

- Every file `log/*/clean/RUNG_learnable_distance_distcosine_q0.75.log` is **cosine distance + percentile gamma `q = 0.75` + SCAD**. It is not cosine with the original fixed MCP `γ = 6`.
- `RUNG_combined` is that same pairing (percentile + cosine, no new parameters). It has no separate log. Treat it as the same cell until a paired run shows otherwise.
- `RUNG_combined_model` adds three scalars: `log_gamma_0`, `raw_decay`, and a blend weight `α`. Logged init is `q = 0.75`, `decay = 0.85`, `α = 0.5`. The gamma at layer `k` is

```text
γ^(k) = sigmoid(α) * (γ0 * decay^k) + (1 - sigmoid(α)) * quantile(y, q)
```

- `RUNG_parametric_gamma` is Euclidean distance plus `γ^(k) = γ0 * 0.85^k`. It does not use cosine.

Code-flow writeups for each class are in `docs/model_codeflow/`. Per-change notes are in `docs/changes/`.

## 4. Measured results

Models in the tables:

- **Base**: `RUNG`, MCP, `γ = 6`
- **Percentile**: `RUNG_percentile_gamma`, Euclidean, `q = 0.75`
- **Cosine+q**: `RUNG_learnable_distance`, cosine, `q = 0.75`
- **Parametric**: `RUNG_parametric_gamma`, Euclidean, decay `0.85`
- **Blend**: `RUNG_combined_model`, cosine + parametric/percentile blend

Clean logs: `log/<data>/clean/`. Attack logs: `log/<data>/attack/`. The same five split seeds are reused, so clean accuracy inside an attack log matches the clean log.

### 4.1 Clean accuracy (%)

| Model | Cora | Citeseer | Pubmed | Chameleon | Cornell | Wisconsin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base MCP γ=6 | 81.3 | 70.8 | 67.6 | 41.5 | 45.2 | 53.1 |
| Percentile q=0.75 | 78.6 | 70.2 | 70.7 | 39.4 | 44.9 | 59.0 |
| Cosine + percentile | 78.0 | 71.3 | **71.3** | **44.3** | **49.9** | 59.2 |
| Parametric decay | 80.6 | **72.9** | 69.5 | 39.4 | 48.7 | **70.4** |
| Blend (combined_model) | **81.9** | 72.9 | 69.1 | 42.5 | 44.1 | 47.4 |

Bold marks the best clean number in that column. Citeseer parametric and blend are tied at 72.9 (72.91 vs 72.88).

Seed noise is small on the citation graphs (typical std 1 to 2 points) and large on WebKB. Wisconsin base clean std is 7.5 points. Cornell base clean std is 5.8 points. A 1-point WebKB movement is inside noise. A 10-point movement is not.

### 4.2 Attacked accuracy at budget 5% (%)

| Model | Cora | Citeseer | Pubmed | Chameleon | Cornell | Wisconsin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base MCP γ=6 | 74.2 | 64.2 | 67.0 | 28.2 | 40.3 | 47.6 |
| Percentile q=0.75 | 76.4 | 67.2 | 70.5 | 40.1 | 44.2 | 57.1 |
| Cosine + percentile | 77.4 | 69.8 | **71.1** | **43.4** | **48.7** | 57.7 |
| Parametric decay | 78.5 | **71.8** | 69.1 | 36.3 | 45.3 | **69.8** |
| Blend | **80.5** | 69.5 | 68.7 | 41.3 | 42.9 | 46.6 |

### 4.3 Attacked accuracy at budget 20% (%)

| Model | Cora | Citeseer | Pubmed | Chameleon | Cornell | Wisconsin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base MCP γ=6 | 64.7 | 55.8 | 64.9 | 20.9 | 35.9 | 40.8 |
| Percentile q=0.75 | 74.8 | 63.4 | 69.6 | 37.4 | 41.6 | 55.1 |
| Cosine + percentile | 77.1 | 68.4 | **70.9** | **42.2** | **49.3** | 57.4 |
| Parametric decay | 73.8 | **70.2** | 68.1 | 32.3 | 44.1 | **68.8** |
| Blend | **78.5** | 66.9 | 68.1 | 40.7 | 41.9 | 47.4 |

### 4.4 Attacked accuracy at budget 100% (%)

| Model | Cora | Citeseer | Pubmed | Chameleon | Cornell | Wisconsin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base MCP γ=6 | 52.9 | 48.4 | 56.9 | 14.5 | 30.3 | 34.2 |
| Percentile q=0.75 | 62.0 | 56.7 | 66.7 | 32.3 | 41.6 | 50.0 |
| Cosine + percentile | 75.4 | 63.5 | **70.3** | **40.5** | **47.6** | 61.2 |
| Parametric decay | 65.3 | **67.3** | 65.8 | 30.1 | 40.7 | **66.7** |
| Blend | **78.2** | 62.0 | 66.7 | 39.8 | 41.8 | 49.5 |

### 4.5 Drop from clean to budget 100% (points)

| Model | Cora | Citeseer | Pubmed | Chameleon | Cornell | Wisconsin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base MCP γ=6 | -28.4 | -22.4 | -10.7 | -27.0 | -14.9 | -18.9 |
| Percentile q=0.75 | -16.6 | -13.5 | -4.0 | -7.1 | -3.3 | -9.0 |
| Cosine + percentile | **-2.6** | -7.8 | **-1.0** | **-3.8** | **-2.3** | +2.0 |
| Parametric decay | -15.3 | **-5.6** | -3.7 | -9.3 | -8.0 | **-3.7** |
| Blend | -3.7 | -10.9 | -2.4 | -2.7 | -2.3 | +2.1 |

The Wisconsin cosine “gain” under a full attack (+2.0) sits inside that run’s seed std (clean std about 4.5 points, attack std about 3.4). Read it as “the attack did not move the mean,” not as a real improvement. Cora parametric at budget 100% has std 14.6 points: one split falls to 37% while another stays at 80%. That schedule is unstable on Cora even when the mean looks acceptable.

### 4.6 Per-dataset champions

Use these as the bars a new run must clear. “Robust” means attacked accuracy at budget 100%.

| Dataset | Family | Clean champion | Robust champion | Clean | Robust @100% |
| --- | --- | --- | --- | ---: | ---: |
| Cora | homophilic citation | Blend | Blend | 81.9 | 78.2 |
| Citeseer | homophilic citation | Parametric (tie on clean) | Parametric | 72.9 | 67.3 |
| Pubmed | homophilic citation | Cosine+q | Cosine+q | 71.3 | 70.3 |
| Chameleon | heterophilic wiki | Cosine+q | Cosine+q | 44.3 | 40.5 |
| Cornell | heterophilic WebKB | Cosine+q | Cosine+q | 49.9 | 47.6 |
| Wisconsin | heterophilic WebKB | Parametric | Parametric | 70.4 | 66.7 |

There is no single family champion. A new global model is judged against this row-wise bar, not against one average.

### 4.7 What was implemented and not measured

| Model | Status in `log/` |
| --- | --- |
| `RUNG_confidence_lambda` | No clean or attack log |
| `RUNG_homophily_adaptive` | No clean or attack log |
| `RUNG_learnable_combined` (`schedule` and `per_layer`) | No clean or attack log |
| `RUNG_new_ADAPTIVE` (inverted MCP in `model/penalty.py`) | No clean or attack log |
| `RUNG_combined` as a distinct run from cosine+percentile | No separate log |
| Projection and bilinear distance | No log (only `distcosine`) |
| Cosine with fixed MCP or fixed SCAD gamma | No log |
| `RUNG_percentile_adv`, `RUNG_parametric_adv` | 7/7 failed in `exp/results/adversarial_training/report.txt` (2026-03-16, about 20 s each, so they died at startup rather than finishing training) |

## 5. How to read the failures

These readings are the filter for the catalog in Section 6. A strategy that fights a reading needs new evidence before it is scheduled.

**Fixed `γ = 6` stops pruning once features smooth.** Early layers have large Euclidean gaps. By layer 10 the gaps are far below 6, so MCP never zeros an edge and the defense is off next to the output. Percentile gamma was built to track that shrinkage. The logs show the side effect: on Cora, Euclidean percentile lowers clean accuracy from 81.3% to 78.6% and still falls to 62.0% at budget 100%. Tracking the distribution is right. Pruning a fixed 25% of edges at every depth is a strong assumption.

**Euclidean distance misses directional attacks.** An edge between two different-class nodes can have a small Euclidean gap when the logits have similar magnitude and a large angle. Cosine distance is `1 - cos`, bounded in `[0, 2]`, and ignores magnitude. That is why cosine+percentile loses 2.6 points on Cora from clean to budget 100%, while base MCP loses 28.4. The same pattern is larger on Chameleon: base falls from 41.5% to 14.5%, cosine+percentile from 44.3% to 40.5%.

**Percentile and cosine are entangled in every strong cosine log.** Until cosine is rerun with a fixed gamma, and until percentile is rerun at other `q`, nobody knows which half of “cosine+q=0.75” produced the flat attack curves. Pubmed percentile alone already reaches 70.7% clean and 66.7% at budget 100%, so percentile is doing real work on that graph. Cora percentile is not enough (62.0% at budget 100% versus 75.4% for cosine+percentile).

**Parametric decay is a heterophily specialist on WebKB and a citation-graph stabilizer on Citeseer.** Wisconsin moves from 53.1% clean / 34.2% robust (base) to 70.4% / 66.7%. Citeseer robust accuracy at budget 100% is 67.3%, the best in that column. The same schedule is the worst clean model on Chameleon (39.4%) and is high-variance on Cora. WebKB and Wikipedia heterophily are different regimes. One “heterophily fix” will not cover both.

**Stacking the winners over-prunes heterophilic graphs.** `docs/changes/018_homophily_adaptive.md` calls this double correction: cosine already flags cross-class edges, and a tight percentile then removes them. The blend logs confirm it. Wisconsin blend clean is 47.4%, below the base 53.1% and far below parametric 70.4%. Cornell blend clean is 44.1%, below the base 45.2% and below cosine+percentile 49.9%. On Cora the same blend is the best model. The blend weight `α` did not learn its way out of the heterophily failure at the logged init.

**Per-layer free gammas on Euclidean features were already abandoned.** `docs/changes/007_learnable_gamma.md` records why. SCAD sends gradient to `γ` only from edges inside the transition band. If a layer starts with every edge below `γ` or every edge above `aγ`, that layer’s threshold never moves. Seed variance was large. Do not bring K independent Euclidean thresholds back.

**The attack used for the tables is structure PGD, not a cosine-aware attacker.** `cosine_attack.py` exists (`docs/changes/017_cosine_attack.md`) and was not the attacker behind these logs. A defense whose score is cosine similarity still needs to be attacked in that same score. Flat curves above are necessary evidence and are not yet sufficient.

**Adversarial training has no accuracy number.** The report lists seven Cora runs, alphas `{0.5, 0.7, 0.9}`, attack frequencies `{1, 5, 10}`, and all of them failed. Diagnose the crash before any claim about curriculum PGD.

## 6. Strategy catalog

Each strategy has an id used again in Section 7. “Change” is relative to `model/rung.py` unless a parent model is named. Pass/fail uses the rule in Section 8.

Cost labels:

- **Run**: existing code, new command.
- **Small**: a new flag or a short branch on an existing class.
- **Medium**: a new forward equation and a fit function.
- **Large**: a new data path (sparse kernels, a new attack loop) or a multi-model router.

### Family A. Disentangle cosine, percentile, and parametric

Do this before any new equation. The logs cannot yet say which factor to keep.

#### S01. Cosine distance with fixed MCP gamma

- **Hypothesis.** The flat attack curves come from the distance, and fixed `γ` can recover the Cora clean accuracy that `q = 0.75` gave up (81.3% base vs 78.0% cosine+q).
- **Change.** Keep the QN-IRLS update. Set `y_ij = 1 - cos(f_i/√d_i, f_j/√d_j)`. Use MCP with a fixed `γ` in `{0.3, 0.5, 0.8, 1.0, 1.5}`. Cosine lives in `[0, 2]`, so `γ = 6` would never prune and is not a candidate.
- **Failure targeted.** Cora clean drop of cosine+q, and the entanglement noted in Section 5.
- **Where it should move.** Up on Cora clean if percentile was over-pruning. Flat attack curves should remain if cosine was the real defense. Hurts Chameleon and Cornell if a global threshold clips legitimate cross-class edges.
- **New parameters.** 0. One discrete `γ`.
- **Cost.** Small, because `DistanceModule` and MCP already exist separately and are not wired this way in the logged runs.
- **Pass.** Cora clean at least 80.5% and Cora budget-100% at least 74%, without dropping Pubmed budget-100% below 68%.

#### S02. Cosine distance with fixed SCAD gamma

- **Hypothesis.** SCAD’s linear transition band is a better match to a bounded distance than MCP’s hard zero, at the same threshold scale.
- **Change.** Same as S01 with differentiable SCAD (`scad_weight_differentiable` in `model/rung_learnable_gamma.py`), `a = 3.7`, `λ_scad = γ / a`, `γ` in `{0.5, 1.0, 1.5}`.
- **Failure targeted.** Same as S01. Also checks whether the logged cosine runs are good because of SCAD rather than because of cosine.
- **Where it should move.** Same datasets as S01. Prefer S02 over S01 if clean accuracy is within 0.5 points and budget-100% is higher.
- **New parameters.** 0.
- **Cost.** Small.
- **Pass.** Same bar as S01.

#### S03. Euclidean percentile `q` sweep

- **Hypothesis.** `q = 0.75` prunes too many homophilic edges and too few or too many heterophilic ones. A higher quantile prunes less.
- **Change.** `RUNG_percentile_gamma` only. Grid `q ∈ {0.50, 0.75, 0.90, 0.95, 0.99}`. `experiments/search_percentile_q.py` already exists and should be the runner.
- **Failure targeted.** Cora Euclidean percentile clean 78.6% vs base 81.3%, and Chameleon percentile budget-100% 32.3% vs cosine+q 40.5%.
- **Where it should move.** `q → 0.9` or `0.95` should help Cora clean. `q → 0.99` is the heterophily-relaxation control (almost no pruning).
- **New parameters.** 0.
- **Cost.** Run.
- **Pass.** Any `q` that beats `q = 0.75` by more than 1 point clean on Cora and does not lose more than 1 point on Pubmed.

#### S04. Cosine percentile `q` sweep

- **Hypothesis.** On a bounded distance, the right prune fraction is not 25%. Homophilic graphs want a high `q`. Heterophilic graphs want `q` near 1.
- **Change.** `RUNG_learnable_distance` mode `cosine`. Same `q` grid as S03.
- **Failure targeted.** Cora clean gap versus the blend (78.0 vs 81.9) and Wisconsin gap versus parametric (59.2 vs 70.4).
- **Where it should move.** High `q` on Cora and Citeseer. Very high `q` on Wisconsin, Cornell, Chameleon.
- **New parameters.** 0.
- **Cost.** Run.
- **Pass.** A single `q` that matches cosine+q=0.75 robustness on Pubmed (budget-100% ≥ 69%) and raises Cora clean above 80%.

#### S05. `lam_hat` sweep

- **Hypothesis.** The skip weight is doing as much work as the penalty. A larger skip (smaller `lam_hat`) protects attacked nodes. A smaller skip (larger `lam_hat`) raises clean accuracy when the graph is trustworthy.
- **Change.** No equation change. Grid `lam_hat ∈ {0.5, 0.7, 0.9, 0.95}` on base MCP and on cosine+q=0.75. Recall `λ = 1/lam_hat - 1`.
- **Failure targeted.** Base collapse under large budgets, where the smoothed state overwrites `F0`.
- **Where it should move.** Smaller `lam_hat` should lift budget-100% on Cora base. Larger `lam_hat` may lift clean accuracy on Pubmed cosine.
- **New parameters.** 0.
- **Cost.** Run.
- **Pass.** Any setting that gains at least 1 point of budget-100% accuracy on Cora or Pubmed at a clean cost of at most 0.5 points.

#### S06. Confirm `RUNG_combined` duplicates cosine+percentile

- **Hypothesis.** `model/rung_combined.py` and cosine mode of `model/rung_learnable_distance.py` implement the same estimator. One paired run is enough to retire a name.
- **Change.** Train both on Cora and Wisconsin for the same splits and the same `q = 0.75`.
- **Failure targeted.** Ambiguous lineage, which makes later ablations easy to misread.
- **Where it should move.** Nowhere, if the hypothesis holds. Means within 0.3 points.
- **New parameters.** 0.
- **Cost.** Run.
- **Pass.** Documented duplicate, or a written difference if the means diverge.

### Family B. The missing high-value cell

#### S07. Cosine plus parametric decay, percentile off

- **Hypothesis.** Cosine supplies the robust distance. Parametric decay supplies a smooth depth schedule and the Wisconsin/Citeseer accuracy. The percentile term in the blend is what over-prunes WebKB.
- **Change.** `y` from cosine. `γ^(k) = γ0 * r^k` with the same parameterization as `model/rung_parametric_gamma.py`, but with `γ0` initialized inside `(0, 2)` (start at 1.0, not at 3.0). No quantile. SCAD or MCP, chosen from S01/S02.
- **Failure targeted.** Wisconsin blend 47.4% vs parametric 70.4%. Cornell blend 44.1% vs cosine+q 49.9%. Cora cosine+q clean 78.0% vs blend 81.9%.
- **Where it should move.** Best shot at a single model that keeps Cora near the blend and Wisconsin near parametric. Risk: Chameleon, where parametric Euclidean already failed (39.4% clean, 30.1% at budget 100%). If Chameleon drops more than 2 points versus cosine+q, this cell is a citation/WebKB model and Chameleon stays on cosine+q.
- **New parameters.** 2 (`log γ0`, `raw_decay`).
- **Cost.** Small. Both pieces exist; the wiring does not.
- **Pass.** Cora clean ≥ 80.5% and budget-100% ≥ 76%. Wisconsin clean ≥ 65%. Pubmed budget-100% ≥ 68%. Chameleon clean within 2 points of 44.3%.

#### S08. Decay-rate grid on S07

- **Hypothesis.** `r = 0.85` was chosen for Euclidean gaps that shrink fast. Cosine gaps do not shrink the same way, so the useful decay is closer to 1.
- **Change.** S07 at `r ∈ {0.70, 0.85, 0.95}` and at a frozen `r = 1` (constant `γ0`, the bridge to S01).
- **Failure targeted.** Cora parametric variance (std 14.6 points at budget 100%), which is consistent with a schedule that decays too far.
- **Where it should move.** `r` near 1 should look like S01. `r = 0.70` should look more like aggressive deep-layer pruning and may hurt heterophily.
- **New parameters.** 2, one of them swept.
- **Cost.** Run, after S07 exists.
- **Pass.** The best `r` beats S07’s default on at least two datasets by 1 point and loses on none by more than 1 clean point.

#### S09. Quantile-banded parametric gamma

- **Hypothesis.** A decay prior is useful, and it should be forbidden from leaving the band of edge gaps that still have a SCAD gradient. That keeps the transition zone populated at every layer, which is the failure mode of free per-layer gammas.
- **Change.** Compute `γ_prior^(k) = γ0 * r^k` and `γ_data^(k) = quantile(y_E, q)`. Set

```text
γ^(k) = median( γ_lo, γ_prior^(k), γ_hi )
```

with `γ_lo = quantile(y_E, 0.5)` and `γ_hi = quantile(y_E, 0.95)`, or clamp `γ_prior` into `[γ_data / a, a * γ_data]`. This is a clamp, not the current blend. The percentile cannot pull `γ` to an extreme; it only rejects impossible values.

- **Failure targeted.** Blend over-pruning on WebKB, and dead gradients when `γ` leaves the data range.
- **Where it should move.** Cora and Citeseer if the prior is right and the data only stabilizes it. Wisconsin if the clamp stops the prior from going to zero.
- **New parameters.** 2, plus fixed band quantiles.
- **Cost.** Medium.
- **Pass.** Matches S07 on Cora and Wisconsin within 1 point and shows non-zero SCAD region-2 mass at every layer (log the fraction).

### Family C. Heterophily-aware weights

RUNG treats a large gap as an attack. On heterophilic graphs a large gap is often the true edge. The repairs below change the weight, not the dataset.

#### S10. Run `RUNG_homophily_adaptive` as it stands

- **Hypothesis.** Nodes with low local homophily should be pruned less. The implemented rule is `q_i = q_base + (1 - h_i) * q_relax`, clipped at `q_max`.
- **Change.** No new code. Grid `q_relax ∈ {0.0, 0.10, 0.20, 0.40}`, `homophily_mode ∈ {from_F0, per_layer}`, `q_max = 0.99`. `q_relax = 0` is the control and should reproduce the parent percentile behavior.
- **Failure targeted.** Double correction on Wisconsin, Cornell, and Chameleon. Documented in `docs/changes/018_homophily_adaptive.md` and confirmed by the blend column.
- **Where it should move.** Wisconsin, Cornell, Chameleon. On Cora, `h_i` is high, so `q_i` stays near `q_base` and the run should stay near cosine+q.
- **New parameters.** 0 learned. Two discrete knobs.
- **Cost.** Run.
- **Pass.** Wisconsin clean ≥ 65% or a gain of at least 5 points over cosine+q (59.2%), with Cora clean ≥ 77%.

#### S11. Inverted MCP penalty (`RUNG_new_ADAPTIVE`)

- **Hypothesis.** Edge-wise soft-label disagreement is a better heterophily signal than a global quantile. The weight already in `PenaltyFunction.homophily_adaptive` is

```text
h_ij = 1 - <p_i, p_j>
W_homo   = max(0, 1/(2y) - 1/(2γ))
W_hetero = max(0, 1/(2γ) - 1/(2y))
W        = (1 - h_ij) W_homo + h_ij W_hetero
```

Homophilic edges keep MCP. Heterophilic edges invert it, so similar features get suppressed and different features are kept.

- **Change.** Run `penalty='adaptive'` / `RUNG_new_ADAPTIVE` on all six datasets at `γ ∈ {1, 3, 6}`.
- **Failure targeted.** Base MCP on Chameleon (budget-100% 14.5%) and Wisconsin (34.2%).
- **Where it should move.** Heterophilic graphs. On Cora it should fall back toward MCP because `h_ij` is small. Risk: early-training `p_i` is noise, so the first epochs can lock in a bad inversion. If that shows up, add a 30-epoch warmup with plain MCP.
- **New parameters.** 0.
- **Cost.** Run.
- **Pass.** Chameleon budget-100% ≥ 35% and Wisconsin clean ≥ 60%, with Cora clean ≥ 79%.

#### S12. Inverted MCP on cosine distance

- **Hypothesis.** S11’s label signal plus S01’s distance is the heterophily version of the Cora winner.
- **Change.** Cosine `y`, then `homophily_adaptive` weights. Fixed `γ` from the S01 winner, or `γ = 1`.
- **Failure targeted.** Cosine+q still trails parametric on Wisconsin by about 11 clean points (59.2 vs 70.4).
- **Where it should move.** Wisconsin and Cornell. Should hold Chameleon, where cosine is already best.
- **New parameters.** 0.
- **Cost.** Small.
- **Pass.** Wisconsin clean ≥ 65% and Chameleon budget-100% ≥ 38%.

#### S13. Inverted MCP on parametric Euclidean gamma

- **Hypothesis.** Wisconsin’s winning schedule is Euclidean parametric. Adding label-aware weights can help Cornell and Chameleon without giving up that schedule.
- **Change.** `RUNG_parametric_gamma` with `W` from `homophily_adaptive` instead of plain MCP/SCAD. Same decay init.
- **Failure targeted.** Parametric collapse on Chameleon (clean 39.4%, budget-100% 30.1%).
- **Where it should move.** Chameleon up, Wisconsin unchanged if `h_ij` is already extreme and the schedule was doing the work.
- **New parameters.** 2, already in the parent.
- **Cost.** Small.
- **Pass.** Chameleon clean ≥ 42% and Wisconsin clean ≥ 68%.

#### S14. Homophily-adaptive `q` on cosine, then on parametric

- **Hypothesis.** S10’s per-node quantile is the right heterophily gate, and it should sit on whichever distance won S01–S07 rather than on the default parent.
- **Change.** Two runs of the S10 rule: one with cosine `y`, one with Euclidean `y` and parametric `γ` replaced by `q_i`. Keep `q_relax = 0.20` first.
- **Failure targeted.** Same double-correction column as S10, after the distance question is settled.
- **Where it should move.** Same as S10, with a higher ceiling if cosine is the right `y`.
- **New parameters.** 0.
- **Cost.** Small, after S10 has a working config.
- **Pass.** Beats S10 on Wisconsin or Chameleon by 2 points and does not lose Cora clean by more than 1 point.

#### S15. Same-class MCP, cross-class keep or negative weight

- **Hypothesis.** A binary decision is stabler than a soft inversion. If the two endpoints’ argmax labels agree, use MCP. If they disagree, set `W_ij = 1` (keep the edge) or `W_ij = -α` (use the edge as a contrast, FAGCN-style).
- **Change.** Inside the weight step only. Start with the keep variant (`W = 1` on disagreeing edges). Try `α ∈ {0.1, 0.3}` only if keep is already better than base on Chameleon.
- **Failure targeted.** Legitimate cross-class edges deleted by MCP and by cosine+percentile.
- **Where it should move.** Chameleon, Cornell, Wisconsin. Cora should be almost unchanged because most edges agree.
- **New parameters.** 0 for the keep variant. 1 scalar if the negative variant is turned on.
- **Cost.** Medium.
- **Pass.** Heterophilic clean mean (average of the three graphs) at least 3 points above cosine+q, Cora clean drop at most 1 point.

#### S16. Two-channel aggregation

- **Hypothesis.** One weight cannot both prune attacks and preserve heterophily. Split the update into a low-pass channel (MCP on predicted-same edges) and a high-pass channel (difference operator on predicted-different edges), then mix them with a scalar `β ∈ (0, 1)`.
- **Change.** Replace the single `W ⊙ Ã` multiply with

```text
F ← (1 - β) * LowPass(F) + β * HighPass(F) + skip(F0)
```

`β` can be a constant 0.5 first, then a single learned sigmoid.

- **Failure targeted.** The single-weight assumption on Chameleon, which no current penalty has solved (best clean is still 44.3%).
- **Where it should move.** Chameleon first. Citation graphs should prefer `β` near 0.
- **New parameters.** 0 or 1.
- **Cost.** Medium.
- **Pass.** Chameleon clean ≥ 48% or budget-100% ≥ 44%, with Cora clean ≥ 80%.

### Family D. Gamma that stays inside a live transition zone

#### S17. Bounded schedule gamma on cosine (`RUNG_learnable_combined`, `schedule`)

- **Hypothesis.** The Euclidean per-layer gamma failed because its scale drifted away from the data. Cosine caps the legal range at 2, and a two-parameter schedule shares gradient across layers. This is the design in `model/rung_learnable_combined.py`:

```text
γ^(k) = sigmoid(raw_g0) * 2 * sigmoid(raw_decay)^k
```

- **Change.** Run the existing class, `gamma_mode=schedule`, on all six datasets.
- **Failure targeted.** Dead transition zones of `RUNG_learnable_gamma`. Also a learned version of S07.
- **Where it should move.** Cora and Pubmed if the init `γ0 = 1`, `r = 0.85` is close. Watch Wisconsin: a learned decay can rediscover the over-pruning of the blend.
- **New parameters.** 2.
- **Cost.** Run.
- **Pass.** Cora budget-100% ≥ 76% and Wisconsin clean ≥ 60%. Log the ten gamma values so a collapsed schedule is visible.

#### S18. Bounded per-layer gamma on cosine (`per_layer` mode)

- **Hypothesis.** With `γ` forced into `(0, 2)`, the transition band stays populated and K free thresholds become trainable. This is the claim in `docs/changes/016_learnable_combined.md`. It is still a risk: K parameters and a narrow band can overfit the five small WebKB training sets.
- **Change.** Existing `gamma_mode=per_layer`. Run on Cora and Wisconsin only. Promote to the other four datasets only if both pass.
- **Failure targeted.** Under-capacity of a two-parameter schedule, if S17 is stable and still short of the champion.
- **Where it should move.** Cora, maybe. Wisconsin is the overfitting check.
- **New parameters.** K (10).
- **Cost.** Run.
- **Pass.** Beats S17 by 1 point on Cora budget-100% with Wisconsin std not larger than the parametric run’s std (about 4.6 clean points).

#### S19. Per-node gamma from local homophily, Euclidean or cosine

- **Hypothesis.** A graph-level `q` is the wrong unit. `q_i(h_i)` in S10 is one implementation. This entry is the parametric analogue: `γ_i^(k) = γ0 * r^k * (1 + c * (1 - h_i))`, so heterophilic nodes get a higher threshold and keep more edges.
- **Change.** One new scalar `c`, initialized at 0 so the model starts as S07 or as parametric Euclidean.
- **Failure targeted.** Global thresholds on mixed neighborhoods (a homophilic node next to a heterophilic node).
- **Where it should move.** Cornell and Wisconsin, whose graphs are small and locally mixed. Less effect on Cora.
- **New parameters.** 1 on top of the schedule.
- **Cost.** Medium.
- **Pass.** Wisconsin or Cornell gain of 3 clean points over the `c = 0` parent, Cora move inside 1 point.

### Family E. Skip connection and what gets smoothed

#### S20. Confidence-weighted lambda, retargeted

- **Hypothesis.** Attacked nodes are the uncertain ones and should trust `F0` more. `model/rung_confidence_lambda.py` already maps `softmax(F0)` to a per-node `λ_i` with a learned sharpness. The parent it was built on is Euclidean learnable-gamma, which is the abandoned branch. The idea is still worth running on the parents that actually win.
- **Change.** First run the existing class once on Cora, as a smoke test that the code trains. Then move the `λ_i` formula onto the Cora champion (blend or S07) and onto parametric Wisconsin. Mode `protect_uncertain` first.
- **Failure targeted.** Uniform skip. Base Cora falls from 74.2% at budget 5% to 52.9% at budget 100% because every node is smoothed the same way.
- **Where it should move.** Large-budget Cora and Citeseer. On Wisconsin the training set is tiny, so a learned sharpness can add variance; treat std as part of the result.
- **New parameters.** 1 (`raw_alpha`) once the gamma parent is fixed.
- **Cost.** Run for the smoke test. Small for the retarget.
- **Pass.** Budget-100% gain of at least 1.5 points on Cora at a clean cost of at most 0.5 points.

#### S21. Layer-wise skip schedule

- **Hypothesis.** Early layers should smooth (small `λ`), late layers should stick to the features (large `λ`), which is a different axis from gamma decay.
- **Change.** `λ^(k) = λ_base * s^k` with `s` fixed in `{0.8, 1.0, 1.2}` on the S07 or blend parent. `s = 1` is the control.
- **Failure targeted.** Over-smoothing at `K = 10` when gamma has already stopped pruning.
- **Where it should move.** Cora clean, if late layers are washing out the MLP.
- **New parameters.** 0 or 1.
- **Cost.** Small.
- **Pass.** Cora clean gain of 0.8 points with budget-100% not worse.

#### S22. Jumping-knowledge readout

- **Hypothesis.** The classifier sees only `F^(K)`. Mid-layer states still carry class signal that later smoothing deletes. A readout `softmax(Σ_k α_k F^(k))` with `α = softmax(learned)` lets the model keep an earlier state.
- **Change.** Store `F` at layers `{0, 4, 9}` and mix them. Propagation itself stays IRLS.
- **Failure targeted.** Chameleon and the deep-layer “defense turns off” problem. Also a cheap way to see whether `K = 10` is past the useful depth.
- **Where it should move.** Heterophilic graphs, where high-frequency signal lives in early layers. Cora should put almost all mass on the last state.
- **New parameters.** 3 mixture logits.
- **Cost.** Medium.
- **Pass.** Chameleon clean ≥ 46% or a Cora clean gain of 0.5 points with the learned `α` logged.

#### S23. Hidden-space distance, class-space aggregation

- **Hypothesis.** `y` is computed in class space, where `C` is 3 to 7. Two different classes can land close together because there is nowhere else to go. Compute cosine distance in the 64-d MLP hidden state, and still aggregate the class logits.
- **Change.** MLP returns `(H, F0)`. `y_ij = 1 - cos(h_i, h_j)`. The QN-IRLS update still runs on `F`. Detach `H` in the first version so the cost stays close to today.
- **Failure targeted.** Cosine on logits saturating (Cora cosine+q clean stuck at 78.0% while the blend, which changes gamma rather than dimension, reaches 81.9%).
- **Where it should move.** Cora and Citeseer clean accuracy. Pubmed if bag-of-words classes overlap in logit space.
- **New parameters.** 0 if `H` is an existing MLP activation. The MLP must expose it.
- **Cost.** Medium.
- **Pass.** Cora clean ≥ 81% and budget-100% ≥ 75%.

#### S24. Propagation early exit

- **Hypothesis.** Many of the 10 steps are idle once `F` has settled, especially under cosine, whose attack curve is already flat. Stopping when `||F^(k+1) - F^(k)|| / ||F^(k)|| < 10^{-3}` preserves the estimator and cuts the dense matmul.
- **Change.** Break the K loop on that test. Log the exit layer.
- **Failure targeted.** Compute, and over-smoothing if late steps still move `F` in a harmful direction. If the exit layer is always 10, the hypothesis is false and `K` stays.
- **Where it should move.** Runtime on Pubmed. Accuracy should match `K = 10` within 0.3 points.
- **New parameters.** 0.
- **Cost.** Small.
- **Pass.** Mean exit layer ≤ 6 on Cora with accuracy inside 0.3 points of the full run.

#### S25. Short un-detached warmup

- **Hypothesis.** Detaching `y` is correct IRLS, and it also prevents the MLP from arranging classes so that cosine can see them. Ten epochs with `y` attached, then the usual detach, lets the representation separate and then returns to the unbiased update.
- **Change.** A flag on the cosine forward. Warmup length `{0, 10, 30}`.
- **Failure targeted.** Cora cosine+q clean gap versus the blend.
- **Where it should move.** Cora and Citeseer. Risk on heterophily: the MLP may learn to push all cross-class pairs apart and the penalty will then delete true edges. Watch Chameleon.
- **New parameters.** 0.
- **Cost.** Small.
- **Pass.** Cora clean gain of 1 point, Chameleon clean drop at most 1 point.

### Family F. Training objective

#### S26. Repair the adversarial trainer, then a narrow rerun

- **Hypothesis.** Curriculum PGD can still help, and the current evidence is a crash, not a negative result. The mixed loss in `docs/changes/013_adversarial_training.md` is `α L_clean + (1-α) L_adv` with budget rising `0.05 → 0.10 → 0.20 → 0.40`.
- **Change.** Read the seven logs under `exp/results/adversarial_training/logs/`, fix the startup failure, and rerun one smoke test for 2 epochs on Cora. Only after that trains: full curriculum on the Cora champion and on Wisconsin parametric. Do not sweep every variant.
- **Failure targeted.** Unmeasured robust training. Also the residual Cora drop of the blend (81.9% clean to 78.2% at budget 100%).
- **Where it should move.** Budget-100% on the model it is attached to. Clean accuracy often falls; the pass rule allows a 1-point clean drop for a 2-point robust gain.
- **New parameters.** 0 architectural. Existing knobs `alpha`, `attack_freq`, `attack_steps`.
- **Cost.** Large, because the trainer is currently red, then Run.
- **Pass.** Smoke test finishes. Full run: budget-100% up by 2 points with clean down by at most 1 point, on the one model that was trained this way.

#### S27. Clean/perturbed consistency

- **Hypothesis.** A lighter robust loss than inner-loop PGD: one random edge-drop or edge-flip view per step, and a KL term between clean logits and perturbed logits, added to cross-entropy. No curriculum.
- **Change.** In the fit function of the surviving model only.
- **Failure targeted.** Same residual robust gap as S26, at much lower cost, and usable even if S26’s trainer stays broken.
- **Where it should move.** Cora and Citeseer at mid budgets (20% to 40%).
- **New parameters.** 1 loss weight.
- **Cost.** Medium.
- **Pass.** Budget-20% gain of 1 point on Cora, clean drop at most 0.5.

### Family G. Evaluation the tables do not contain

#### S28. Cosine adaptive attack on the winners

- **Hypothesis.** Part of the flat cosine curve is an attacker that does not optimize the defense’s own score. `cosine_attack.py` adds a stealth term so inserted edges look small in cosine distance.
- **Change.** No training change. Attack base, cosine+q, parametric, and the blend at budgets `{0.05, 0.20, 1.00}` with the default `β = 1` and with `β = 2`.
- **Failure targeted.** Evaluation gap in Section 5.
- **Where it should move.** If cosine+q falls toward base under this attack, Family A’s “cosine is the defense” claim is weaker and gamma/lambda work matters more. If it stays flat, the claim holds.
- **New parameters.** 0.
- **Cost.** Run.
- **Pass.** This is a measurement, not a model. Record it before Wave 4 declares a winner.

#### S29. Estimation bias next to accuracy

- **Hypothesis.** Accuracy can rise while the estimator becomes biased, which is the failure RUNG was written to avoid. `docs/changes/003_estimation_bias_metrics.md` and `RUNG.get_aggregated_features` already support the metric.
- **Change.** Report bias on clean graphs and at budget 20% for every model that passes a wave.
- **Failure targeted.** Selecting a model that memorizes labels through a biased penalty (plain L1, or a negative-weight channel that is no longer unbiased).
- **Where it should move.** Diagnostic. A model that gains accuracy and doubles bias is kept for accuracy and flagged.
- **New parameters.** 0.
- **Cost.** Run.
- **Pass.** The number is in the wave report.

#### S30. Same-split baselines

- **Hypothesis.** Gains versus base RUNG are real, and gains versus GCN, GAT, APPNP, and SoftMedian on these exact splits are the ones that matter for a paper table. Those baselines are already constructable from `exp/config/get_model.py` and have no paired log in this result set.
- **Change.** Train and attack GCN, GAT, APPNP, and SoftMedian on the six datasets at the same budgets.
- **Failure targeted.** Missing external reference.
- **Where it should move.** Context only.
- **New parameters.** 0.
- **Cost.** Run.
- **Pass.** A table exists, same splits, same budgets.

### Family H. Compute, same estimator

#### S31. Edge-only distance and weights

- **Hypothesis.** The update uses `(W ⊙ A)`. Computing `y` for non-edges does not change `F` for MCP, SCAD, L1, or L2. Percentile gamma only needs the edge set too, if the quantile is taken over edges, which is what the percentile docs already specify (`γ = quantile({y_ij : (i,j) ∈ E})`).
- **Change.** Build `y` from edge endpoints, scatter weights back to a sparse matrix, and replace the dense matmul with a sparse one. Match dense `F` on Cora to a numerical tolerance before any accuracy run.
- **Failure targeted.** Pubmed runtime and the impossibility of ogbn-arxiv at `N^2`.
- **Where it should move.** Wall clock. Accuracy unchanged.
- **New parameters.** 0.
- **Cost.** Large.
- **Pass.** Cora logits match the dense model within `1e-4` relative error, and Pubmed step time drops.

#### S32. Step count under cosine

- **Hypothesis.** Cosine curves are flat because the metric separates classes early. `K ∈ {2, 5, 10}` on cosine+q and on S07 will show whether the last five steps buy accuracy.
- **Change.** `prop_step` only.
- **Failure targeted.** Fixed `K = 10` copied from the Euclidean paper setting.
- **Where it should move.** If `K = 5` matches `K = 10`, every later run gets cheaper.
- **New parameters.** 0.
- **Cost.** Run.
- **Pass.** The shortest `K` within 0.3 clean points and 0.5 budget-100% points of `K = 10`, on Cora and Pubmed.

#### S33. Mixed precision on the dense matmul

- **Hypothesis.** The pairwise gemm is memory-bound. FP16 for `y` and FP32 for the QN update keeps the estimator and cuts Pubmed memory.
- **Change.** Autocast around the distance only, after S31 or before it if S31 slips.
- **Failure targeted.** Pubmed and any larger graph still forced to stay dense.
- **Where it should move.** Memory and step time. Watch SCAD region masks, which are sensitive to ties.
- **New parameters.** 0.
- **Cost.** Small once a profile exists.
- **Pass.** Cora agreement with FP32 within 0.3 accuracy points and a measured memory drop on Pubmed.

### Family I. One model across regimes

#### S34. Homophily router

- **Hypothesis.** The logs already picked the model per regime. A router is more honest than another three-way blend. Estimate edge homophily from the training labels (or from `F0` when labels are sparse). If it is high, run the Cora stack (blend, or S07 if that wins). If it is low and the graph is small and WebKB-like (Cornell, Wisconsin, Texas, Actor), run parametric Euclidean. If it is low and Wikipedia-like (Chameleon, Squirrel), run cosine+percentile with the S04 `q`.
- **Change.** A dispatcher in front of existing forwards. The first version can be a dataset rule. The second version is a threshold on measured homophily, tuned on the validation split.
- **Failure targeted.** The blend’s Wisconsin collapse and parametric’s Chameleon collapse. Both come from forcing one mechanism.
- **Where it should move.** The weak dataset of each specialist, by handing that dataset to the other specialist.
- **New parameters.** 0 inside the experts. One threshold if the rule is learned from homophily.
- **Cost.** Medium.
- **Pass.** Matches each row of the champion table in Section 4.6 within 1 clean point and within 1 budget-100% point.

## 7. Combinations

The full product of distance, gamma, penalty, lambda, and loss is a few hundred runs. The cells below are the ones the logs justify. Everything else waits until the wave that owns it has a winner.

### 7.1 Distance by gamma, the core grid

|  | Fixed γ | Percentile q | Parametric decay | Blend | Bounded schedule |
| --- | --- | --- | --- | --- | --- |
| Euclidean MCP/SCAD | **Logged base.** Cora clean leader among simple models, weak under attack. | **Logged.** Helps Pubmed, hurts Cora clean. Sweep q (S03). | **Logged.** Wisconsin and Citeseer champion. High variance on Cora. | Not planned. The logged blend already uses cosine. | Not planned on Euclidean. Dead-zone failure is already documented. |
| Cosine | **Missing. S01, S02.** Highest-priority ablation. | **Logged as learnable_distance and as RUNG_combined.** Pubmed, Cornell, Chameleon champion. Sweep q (S04). | **Missing. S07, S08.** Highest-priority new cell. | **Logged combined_model.** Cora champion. Hurts Wisconsin and Cornell. | **Implemented, unlogged. S17.** |

### 7.2 Penalty by parent

| Parent | Plain MCP/SCAD | Inverted MCP (S11–S13) | Per-node q (S10, S14) | Same-class / cross-class (S15) |
| --- | --- | --- | --- | --- |
| Euclidean fixed γ | Logged base | S11, unlogged code | Skip until S10 shows a signal | S15 if S11 is weak |
| Euclidean parametric | Logged Wisconsin winner | S13 | S14 second branch | Only if S13 misses Chameleon |
| Cosine + fixed or parametric | S01, S07 | S12 | S14 first branch | S15 on the cosine parent |

### 7.3 What may be stacked, and what may not

Allowed stacks, in order, each conditional on the previous pass:

1. S07 (cosine + parametric) is the primary stack. It is two logged winners with the harmful term removed.
2. S07 plus S10’s `q_i(h_i)` only on nodes the schedule still over-prunes. One heterophily gate, not a second global prune.
3. The winner of (1)–(2) plus S20’s `λ_i`. Skip strength is a different axis from the penalty.
4. That model plus S22 or S23 if clean accuracy is still short of the Cora blend.
5. S26 or S27 last, on one frozen architecture.

Stacks that the logs already argue against:

- Percentile plus parametric plus cosine on WebKB. That is the logged blend. Wisconsin clean fell by 23 points versus parametric.
- K independent Euclidean gammas. Documented dead gradients.
- Projection or bilinear distance before S01 finishes. `docs/changes/011_learnable_distance.md` says to try them only after cosine’s gain is understood. Cosine is still entangled with percentile, so the gain is not understood yet.
- Adversarial training on every variant. The trainer failed seven times, and the compute is large. One parent, after the crash is fixed.
- A third learned blend weight. `α` was supposed to turn the percentile term off on heterophilic graphs and did not, at the logged initialization and learning rate.

### 7.4 Named combination list

Use these ids in experiment folders so a result can be traced to one row.

| Id | Combination | Wave |
| --- | --- | --- |
| S01 | Cosine + fixed MCP γ grid | 1 |
| S02 | Cosine + fixed SCAD γ grid | 1 |
| S03 | Euclidean percentile q grid | 0 |
| S04 | Cosine percentile q grid | 0 |
| S05 | lam_hat grid on base and on cosine+q | 0 |
| S06 | Duplicate check, RUNG_combined vs cosine+q | 0 |
| S07 | Cosine + parametric, no percentile | 1 |
| S08 | S07 decay grid | 1 |
| S09 | Parametric prior clamped to a quantile band | 1 |
| S10 | Homophily-adaptive q, existing class | 0 |
| S11 | Inverted MCP, existing class | 0 |
| S12 | Inverted MCP + cosine | 2 |
| S13 | Inverted MCP + parametric Euclidean | 2 |
| S14 | Per-node q on the Wave 1 winner | 2 |
| S15 | Same-class MCP, cross-class keep | 2 |
| S16 | Two-channel low-pass / high-pass | 2 |
| S17 | Cosine + bounded decay schedule | 0 |
| S18 | Cosine + K bounded gammas | 0, gated |
| S19 | Per-node parametric gamma | 2 |
| S20 | Confidence λ on the survivor | 3 |
| S21 | Layer-wise λ schedule | 3 |
| S22 | Jumping-knowledge readout | 3 |
| S23 | 64-d cosine, class-space aggregation | 3 |
| S24 | Early exit inside K | 3 |
| S25 | Un-detached warmup | 3 |
| S26 | Curriculum PGD after the trainer is fixed | 4 |
| S27 | Clean/perturbed consistency | 4 |
| S28 | Cosine adaptive attack | 0 and again at 4 |
| S29 | Estimation bias | every wave |
| S30 | GCN, GAT, APPNP, SoftMedian | 0 |
| S31 | Sparse edge-only y and W | 5 |
| S32 | K grid | 0 on cosine+q, repeat on the survivor |
| S33 | FP16 distance | 5 |
| S34 | Homophily router over the winning experts | 2 |

## 8. Experiment order

```text
Wave 0   existing code, no new equations
           S03 S04 S05 S06 S10 S11 S17 S18 S28 S30 S32
Wave 1   disentangle the winner
           S01 S02 S07 S08 S09
Wave 2   heterophily gate, then the router
           S12 S13 S14 S15 S16 S19 S34
Wave 3   skip, readout, representation, on one parent
           S20 S21 S22 S23 S24 S25
Wave 4   training loss, then the adaptive attack again
           S26 S27 S28
Wave 5   same estimator, lower cost
           S31 S32 S33
```

Wave 0 can run immediately. Wave 1 should start as soon as S01’s wiring exists; it does not need to wait for the whole of Wave 0, and it should not wait for Wave 2. Wave 2 consumes the Wave 1 winner as its cosine parent. Wave 3 touches one model. Wave 4 starts only after a Cora smoke test of the trainer prints a real validation accuracy. Wave 5 is independent and can proceed in parallel once S31’s numerical check is the goal.

Inside a wave, prefer Cora plus Wisconsin as the screen (one homophilic champion, one heterophilic champion, opposite failure modes). Promote a screened winner to Citeseer, Pubmed, Chameleon, and Cornell. Texas, Actor, and Squirrel are the hold-out heterophilic graphs after the six-dataset table is stable. ogbn-arxiv waits for S31.

## 9. Protocol and pass rule

Match the existing logs so a new number can sit in Section 4 without a footnote.

- Splits: the five splits from `train_eval_data/get_dataset.py`.
- Epochs: 300, same early stopping as the current fit functions.
- Attack: structure PGD, budgets `0.05, 0.10, 0.20, 0.40, 1.00`, the same step count as `run_all.py` (100).
- Metrics: clean accuracy, attacked accuracy, and the drop. Mean and std over the five splits. For any gamma variant, also log per-layer `γ` and the fraction of edges in the SCAD transition band.
- Screen pair: Cora and Wisconsin. Full six only after the screen passes.

A strategy **passes** when all of the following hold.

1. On at least one dataset it beats that row’s champion in Section 4.6 by more than the seed noise: **1.0 point** on Cora, Citeseer, and Pubmed, **3.0 points** on Chameleon, Cornell, and Wisconsin. WebKB std is several points, so a 1-point WebKB gain is not a result.
2. It does not lose more than **1.0 clean point** on any homophilic dataset versus that dataset’s clean champion, and not more than **3.0 clean points** on any heterophilic dataset versus that dataset’s clean champion.
3. Budget-100% accuracy does not fall by more than those same tolerances versus the robust champion of that dataset.
4. The run finished five splits. A crash is not a negative scientific result; it is an open bug, as with the seven adversarial-training failures.

A strategy that wins one family and breaks the other can still be kept as an expert for S34. It does not replace the global champion, and it does not get stacked further in Wave 3.

Report template for each id, so later docs stay comparable:

```text
id, parent, datasets, clean mean±std, attack@0.05, attack@0.20, attack@1.00,
clean drop vs champion, robust drop vs champion, pass/fail, one-line reason
```

## 10. Leave these alone

- **K independent Euclidean gammas.** The transition-zone gradient failure is written up in `docs/changes/007_learnable_gamma.md` and was the reason parametric and percentile replaced that model.
- **Another blend of percentile and parametric on top of cosine for WebKB.** The logged blend is that experiment. It is the Cora champion and the Wisconsin regression. Repeating it at the same `q = 0.75`, `decay = 0.85`, `α = 0.5` will reproduce Section 4.
- **Projection and bilinear distance as the next default.** They add parameters in a space whose scale percentile gamma would have to relearn. Schedule them only if S01 and S23 both plateau at least 1 point short of the Cora blend.
- **Adversarial training as a default loop.** Turn it on for one parent after S26’s smoke test prints validation accuracy.
- **Dense all-pairs `y` as the long-term implementation.** It is the right reference for numerical checks and the wrong implementation for Pubmed-scale and larger graphs. S31 is the replacement, after the estimator is chosen.

## 11. Suggested first commands

Wave 0 uses flags the repo already documents in `run_all.py` and the change notes. Wave 1 needs the S01/S07 wiring before these exact commands exist; they are the intended interface.

```bash
# Screen the primary new cell on Cora and Wisconsin
python run_all.py --datasets cora wisconsin --models S07

# Wave 0 models that were already implemented, plus the new preset ids
python run_all.py --datasets cora wisconsin --models S10 S11 S17 S18

# Sweep a preset knob. Historical defaults are left alone; other values override.
python run_all.py --datasets cora wisconsin --models S01 --gamma 0.5
python run_all.py --datasets cora wisconsin --models S04 --percentile_q 0.9
python run_all.py --datasets cora wisconsin --models S05 --lam_hat 0.5

# Cosine-aware attack of a strategy (S28)
python cosine_attack.py --model S28 --data cora

# Baselines named by S30
python run_all.py --datasets cora wisconsin --models GCN GAT APPNP
```

`python experiments/run_strategies.py --list` prints every id. Preset hyperparameters live in `model/strategy_presets.py`. `--model S07` is the training entry. `--gamma 6` is the historical MCP default and does not overwrite a preset; any other `--gamma` does.

## 12. File map

| Question | File |
| --- | --- |
| Base update | `model/rung.py` |
| MCP, SCAD, L1, L2, inverted weight | `model/penalty.py` |
| Percentile gamma | `model/rung_percentile_gamma.py` |
| Parametric decay | `model/rung_parametric_gamma.py` |
| Cosine / projection / bilinear | `model/rung_learnable_distance.py` |
| Cosine + percentile, no new scalars | `model/rung_combined.py` |
| Cosine + bounded learnable gamma | `model/rung_learnable_combined.py` |
| Cosine + blend | `model/rung_combined_model.py` |
| Per-node quantile | `model/rung_homophily_adaptive.py` |
| Per-node lambda | `model/rung_confidence_lambda.py` |
| Curriculum attack loss | `train_eval_data/adversarial_trainer.py` |
| Failed adv-training report | `exp/results/adversarial_training/report.txt` |
| Cosine-aware attacker | `cosine_attack.py` |
| Clean and attack numbers | `log/<dataset>/clean/`, `log/<dataset>/attack/` |
| Per-variant code flow | `docs/model_codeflow/` |
| Per-change notes | `docs/changes/` |
