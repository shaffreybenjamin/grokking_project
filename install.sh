#!/bin/bash
# Setup script for the mathematics_thesis environment.
# Detects GPU availability and installs the appropriate PyTorch build,
# then installs all other dependencies from requirements.txt.

set -e  # Exit immediately if any command fails

echo "==> Checking environment..."

# Verify we're in the right conda env
if [[ "$CONDA_DEFAULT_ENV" != "mathematics_thesis" ]]; then
    echo "ERROR: This script must be run inside the 'mathematics_thesis' conda env."
    echo "Run: conda activate mathematics_thesis"
    exit 1
fi

# Verify pip points to the conda env (not system pip)
if ! pip --version | grep -q "mathematics_thesis"; then
    echo "WARNING: pip does not appear to be from the conda env."
    echo "         Continuing, but check 'which pip' if you hit issues."
fi

# Detect GPU
echo "==> Detecting GPU..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "    NVIDIA GPU detected. Installing CUDA-enabled PyTorch."
    pip install torch torchvision
else
    echo "    No GPU detected. Installing CPU-only PyTorch."
    pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple
fi

# Install everything else
echo "==> Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "==> Configuring nbstripout..."
nbstripout --install

# Verify
echo ""
echo "==> Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device:     {torch.cuda.get_device_name(0)}')
"

echo ""
echo "==> Setup complete."