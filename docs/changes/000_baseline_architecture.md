# 000 — Baseline Architecture Documentation

## Date
2026-02-27

## Actual Repository Structure
> Note: The actual structure differs from the assumed scaffold in the task prompt.
> The codebase does NOT use `models/`, `utils/`, `main.py` — it uses the structure below.

```
RUNG_ML_Project/
├── model/                   # Model definitions
│   ├── rung.py              # Main RUNG model — PRIMARY FILE
│   ├── att_func.py          # Edge weight (attention) function factories
│   ├── rho.py               # Raw rho/w functions (MCP, L1)
│   ├── mlp.py               # MLP backbone
│   ├── gcn.py               # GCN baseline
│   ├── gat.py               # GAT baseline
│   └── softmedian.py        # SoftMedian baseline
├── train_eval_data/
│   ├── get_dataset.py       # Dataset loading (cora, citeseer, heterophilic)
│   └── fit.py               # Training loop (Adam + cross-entropy)
├── exp/
│   ├── config/
│   │   ├── get_model.py     # Model factory for all datasets (PRIMARY CONFIG)
│   │   ├── get_model_cora.py
│   │   └── get_model_citeseer.py
│   └── result_io.py         # Save/load results and models
├── gb/                      # Graph-benchmark utility library
│   ├── attack/gd.py         # PGD attack implementation
│   ├── metric.py            # Evaluation metrics (accuracy, margin)
│   └── pert.py              # Perturbation utilities
├── data/
│   ├── cora.npz             # Cora ML dataset
│   └── citeseer.npz         # Citeseer dataset
├── utils.py                 # Shared utilities (add_loops, sym_norm, pairwise_euclidean)
├── clean.py                 # Training entry point
└── attack.py                # Attack evaluation entry point
```

## Core Files

### `model/rung.py` — Main RUNG Model
Implements RUNG as a decoupled GNN (APPNP-style):
1. MLP maps raw features X → F⁰ (class-space initial features)
2. QN-IRLS propagation runs K iterations of robust graph smoothing
3. Returns final F^(K) as logits (no separate classification head)

Key design: edge weight function `w_func` is passed as a callable,
enabling penalty-agnostic propagation.

### `model/att_func.py` — Edge Weight Factories
- `get_mcp_att_func(gamma)` → MCP edge weight callable
- `get_scad_att_func(lam_att, gamma)` → SCAD edge weight callable
- `get_l12_att_func(norm)` → L1 or L2 edge weight callable (used for APPNP/RUNG-L1)

### `model/rho.py` — Low-level Penalty Functions
- `w_l1(y)` → 1/(2y)
- `w_ruge(y, gamma)` → max(0, 1/(2y) - 1/(2*gamma))  [MCP]
- `get_w_ruge(gamma)` → factory

### `train_eval_data/get_dataset.py` — Dataset Loading
- Loads cora/citeseer from `.npz` files in `data/`
- Returns `(A, X, y)` as dense float32 tensors
- `A`: [N, N] adjacency, `X`: [N, F] features, `y`: [N] labels
- `get_splits(y)` → 5 deterministic 10-10-80 train/val/test splits
- Has an `else` branch for heterophilic datasets from `data/heter_data/{name}/`

### `train_eval_data/fit.py` — Training Loop
- Adam optimizer with cross-entropy loss
- Calls `model(A, X)` which returns logits
- Prints validation accuracy every 10 epochs

### `exp/config/get_model.py` — Model Factory
Maps `model_name` + `custom_model_params` → `(model, fit_params)`.
Uses `custom_model_params['gamma']` and `custom_model_params['norm']` to select
penalty and gamma threshold.

### `utils.py` — Shared Utilities
- `add_loops(A)`: A + I
- `sym_norm(A)`: D^{-1/2} A D^{-1/2}
- `pairwise_squared_euclidean(X, Y)`: efficient ||X_i - Y_j||² for all pairs
- `accuracy(scores, y_true)`: standard classification accuracy

## Original Model Forward Pass

```python
def forward(self, A, F):
    # decoupled architecture: F = RUNG(MLP(A, F0))
    F0 = self.mlp(F)

    A = add_loops(A)           # add self-loop to avoid zero degree
    D = A.sum(-1)              # [N] degree vector
    D_sq = D.sqrt().unsqueeze(-1)
    A_tilde = sym_norm(A)      # D^{-1/2} A D^{-1/2}

    F = F0                     # initialize with MLP output

    for layer_number in range(self.prop_layer_num):
        # Z_{ij} = ||f_i/sqrt(d_i) - f_j/sqrt(d_j)||_2^2
        Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
        # W_{ij} = d_{y^2} rho(y),  y = ||f_i - f_j||_2
        W = self.w(Z.sqrt())
        W[torch.arange(W.shape[0]), torch.arange(W.shape[0])] = 0  # zero diagonal
        W[torch.isnan(W)] = 1  # NaN guard

        if self.quasi_newton:  # QN-IRLS (default)
            Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)
            F = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat
        else:                  # IRLS (with manual stepsize eta)
            diag_q = torch.diag((W * A).sum(-1)) / D
            grad_smoothing = 2 * (diag_q - W * A_tilde) @ F
            grad_reg = 2 * (self.lam * F - self.lam) * F0
            F = F - self.eta * (grad_smoothing + grad_reg)

    return F
```

## Original Hyperparameters

| Parameter    | Default | Description                                     |
|-------------|---------|--------------------------------------------------|
| gamma        | 6.0     | MCP threshold (from CLI --gamma)                |
| lam_hat      | 0.9     | Skip-connection strength (λ = 1/lam_hat - 1)   |
| prop_step    | 10      | Number of QN-IRLS propagation layers            |
| hidden_dims  | [64]    | MLP hidden layer sizes                          |
| dropout      | 0.5     | Dropout rate in MLP                             |
| lr           | 5e-2    | Adam learning rate                              |
| weight_decay | 5e-4    | L2 regularization                              |
| max_epoch    | 300     | Training epochs                                 |

## Penalty / Norm Convention

The `w_func` callable in RUNG computes:
```
W_ij = d_{y²} ρ(y)  evaluated at  y = ||f_i/√d_i - f_j/√d_j||_2
```
Note: The convention in this codebase is `W = 1/(2y) - 1/(2*gamma)` clamped to [0, max_z],
which is the derivative of MCP w.r.t. y².

## Datasets Supported

| Dataset   | Source        | Nodes  | Features | Classes | Notes           |
|-----------|---------------|--------|----------|---------|-----------------|
| cora      | data/cora.npz | ~2485  | 1433     | 7       | Cora ML (LCC)   |
| citeseer  | data/citeseer.npz | ~2120 | 3703  | 6       | Citeseer (LCC)  |

Heterophilic datasets supported via `data/heter_data/{name}/` (requires pre-downloading).

## Baseline Results (from paper Table 1, Cora ML local attack)

| Budget | RUNG Accuracy |
|--------|---------------|
| 0%     | 84.0 ± 5.3    |
| 20%    | 75.3 ± 6.9    |
| 50%    | 72.7 ± 8.5    |
| 100%   | 70.7 ± 10.6   |
| 150%   | 69.3 ± 9.8    |
| 200%   | 69.3 ± 9.0    |

## Entry Points

```bash
# Train clean model:
python clean.py --model='RUNG' --norm='MCP' --gamma=6.0 --data='cora'

# PGD attack on trained model:
python attack.py --model='RUNG' --norm='MCP' --gamma=6.0 --data='cora'
```

## CLI Arguments (`clean.py` / `attack.py`)

| Arg           | Default | Choices             | Description               |
|--------------|---------|---------------------|---------------------------|
| --model       | RUNG    | RUNG, GCN, GAT, ... | Model architecture        |
| --norm        | MCP     | MCP, L1, L2, SCAD   | Penalty type for RUNG     |
| --gamma       | 6.0     | float               | Penalty threshold         |
| --data        | cora    | cora, citeseer, ... | Dataset name              |
| --lr          | 5e-2    | float               | Learning rate             |
| --weight_decay| 5e-4    | float               | Weight decay              |
| --max_epoch   | 300     | int                 | Training epochs           |
