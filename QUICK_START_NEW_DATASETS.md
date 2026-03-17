# Quick Start: New Datasets Support

This guide shows how to train and evaluate the 4 core RUNG models on the 3 new datasets.

## ✅ Supported New Datasets

| Dataset | Type | Nodes | Auto-Download | Cache Location |
|---------|------|-------|----------------|-----------------|
| **pubmed** | Citation (homophilic) | ~19K | Pre-cached `.pt` | `data/pubmed/` |
| **chameleon** | WikipediaNetwork (heterophilic) | ~2.3K | Yes (torch_geometric) | `data/heter_data/chameleon/` |
| **ogbn-arxiv** | OGB Citation | ~169K | Yes (OGB) | `data/ogb/ogbn-arxiv/` |

Plus the original datasets: **cora**, **citeseer**

---

## 🚀 Basic Usage

### 1. **Smoke Test** (quick verification, no PGD attack)
```bash
python run_all.py \
  --datasets cora pubmed chameleon \
  --models RUNG \
  --max_epoch 10 \
  --skip_attack
```

### 2. **Train + Evaluate 4 Core Models on 2 Datasets**
```bash
python run_all.py \
  --datasets cora pubmed \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined \
  --max_epoch 50
```

### 3. **Test Single Dataset with All 4 Models**
```bash
python run_all.py \
  --datasets chameleon \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined
```

### 4. **Train Only (skip PGD attack)**
```bash
python run_all.py \
  --datasets ogbn-arxiv \
  --models RUNG \
  --skip_attack
```

### 5. **All New Datasets with Full Evaluation**
```bash
python run_all.py \
  --datasets pubmed chameleon ogbn-arxiv \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined \
  --max_epoch 300
```

---

## 📋 Output Locations

Logs are saved to:
```
log/<dataset>/clean/<model>_<norm>_<gamma>.log
log/<dataset>/attack/<model>_norm<norm>_gamma<gamma>.log
```

For example:
```
log/pubmed/clean/RUNG_MCP_6.0.log
log/chameleon/attack/RUNG_combined_normMCP_gamma6.0.log
log/ogbn-arxiv/clean/RUNG_percentile_gamma_MCP_6.0.log
```

---

## ⚙️ Advanced Configuration

### Percentile Gamma Settings (for RUNG_percentile_gamma, RUNG_learnable_distance, RUNG_combined)
```bash
python run_all.py \
  --datasets pubmed chameleon \
  --models RUNG_percentile_gamma RUNG_combined \
  --percentile_q 0.70  # Lighter pruning (default: 0.75)
```

### Learnable Distance Mode (for RUNG_learnable_distance)
```bash
python run_all.py \
  --datasets ogbn-arxiv \
  --models RUNG_learnable_distance \
  --distance_mode cosine  # Options: cosine, projection, bilinear
```

### Custom Attack Budgets
```bash
python run_all.py \
  --datasets cora \
  --models RUNG \
  --budgets 0.05 0.10 0.20 0.50 1.00  # Custom budget list
```

---

## 📦 Dependencies

For the new datasets to auto-download, ensure you have:
```bash
pip install torch_geometric ogb
```

If not installed, run_all.py will fail with a helpful error message pointing you to install them.

---

## 🔍 Troubleshooting

### "ModuleNotFoundError: No module named 'ogb'"
```bash
pip install ogb
```

### "ModuleNotFoundError: No module named 'torch_geometric'"
```bash
pip install torch_geometric
```

### "Connection error downloading ogbn-arxiv"
- The dataset will be downloaded on first run (may take a few minutes)
- Subsequent runs load from cache at `data/ogb/ogbn-arxiv/`
- Check your internet connection and disk space

### "CUDA out of memory"
- Try reducing batch size or using `--skip_attack` for training only
- Use `--max_epoch 100` instead of 300 for faster testing

---

## 📊 Viewing Results

After running, visualize all results:
```bash
python plot_logs.py
```

This generates comparison plots for all models across all datasets.

---

## 4️⃣ The 4 Core Models Explained

| Model | Method | Learnable Params |
|-------|--------|------------------|
| **RUNG** | Fixed penalty (MCP, gamma=6.0) | No new params |
| **RUNG_percentile_gamma** | Adaptive gamma from data quantiles | No new params |
| **RUNG_learnable_distance** | Flexible distance metric + percentile gamma | Yes (distance module) |
| **RUNG_combined** | Cosine distance + percentile gamma | No new params |

All 4 models should run fine on all 5 datasets with the updated code.

---

## 📝 Example: Full Comparison Run

```bash
# Train all 4 models on all 5 datasets with extended budgets
python run_all.py \
  --datasets cora citeseer pubmed chameleon ogbn-arxiv \
  --models RUNG RUNG_percentile_gamma RUNG_learnable_distance RUNG_combined \
  --max_epoch 300 \
  --budgets 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
```

This will:
1. Train all 4 models for 300 epochs on all 5 datasets
2. Evaluate robustness against PGD attacks at 11 different perturbation budgets
3. Save logs to `log/*/clean/` and `log/*/attack/`
4. Generate comparison plots with `python plot_logs.py`

---

**Happy training! 🎉**
