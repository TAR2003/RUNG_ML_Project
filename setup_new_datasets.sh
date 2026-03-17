#!/bin/bash
# setup_new_datasets.sh - Install and verify new dataset support

set -e

echo "============================================================"
echo "Setting up new datasets support for RUNG ML Project"
echo "============================================================"

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check PyTorch
echo ""
echo "Checking PyTorch..."
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')" || {
    echo "✗ PyTorch not found. Please install it first:"
    echo "  pip install torch"
    exit 1
}

# Install torch_geometric and ogb
echo ""
echo "Installing torch_geometric and ogb packages..."
pip install torch_geometric ogb --quiet

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch_geometric; print(f'✓ torch_geometric version: {torch_geometric.__version__}')"
python -c "from ogb.nodeproppred import NodePropPredDataset; print('✓ ogb installed')"

echo ""
echo "============================================================"
echo "✓ Setup complete! You can now use:"
echo "  - pubmed dataset"
echo "  - chameleon dataset"
echo "  - ogbn-arxiv dataset"
echo ""
echo "Try running:"
echo "  python run_all.py --datasets pubmed --models RUNG --max_epoch 5"
echo "============================================================"
