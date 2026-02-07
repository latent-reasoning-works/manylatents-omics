#!/bin/bash
# Setup DNA encoder venv (Evo2 + ESM3)
#
# IMPORTANT: The prebuilt transformer-engine wheels have specific torch version requirements.
# As of v2.11, wheels are built for torch 2.8.0+cu129 (but cu126 works with manual install).
#
# Usage: source scripts/setup-dna-venv.sh
# Must be run with CUDA module loaded (cuda/12.5.0)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load CUDA if not already loaded
module load cuda/12.5.0 2>/dev/null || true

# Create venv if needed
if [ ! -d .venv-dna ]; then
    echo "Creating DNA venv..."
    uv venv .venv-dna --python 3.12
fi

export VIRTUAL_ENV="$PROJECT_DIR/.venv-dna"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Step 1: Install torch 2.8.0+cu126 (compatible with transformer-engine prebuilt wheels)
echo "Installing torch 2.8.0+cu126..."
uv pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu126

# Step 2: Download and install transformer-engine wheel from GitHub
# The wheel has a non-standard version string, so we rename it for uv compatibility
TE_WHEEL_URL="https://github.com/NVIDIA/TransformerEngine/releases/download/v2.11/transformer_engine_torch-2.11.0%2Bcu12torch2.8.0%2Bcu129cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
TE_WHEEL="/tmp/transformer_engine_torch-2.11.0-cp312-cp312-linux_x86_64.whl"

if [ ! -f "$TE_WHEEL" ]; then
    echo "Downloading transformer-engine wheel..."
    curl -sL "$TE_WHEEL_URL" -o "$TE_WHEEL"
fi
echo "Installing transformer-engine..."
uv pip install "$TE_WHEEL"

# Step 3: Install wheel (needed for flash-attn build)
uv pip install wheel

# Step 4: Install flash-attn (builds from source, needs CUDA)
echo "Building flash-attn (this may take a few minutes)..."
export CUDA_HOME=${CUDA_PATH:-/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/12.5.0}
uv pip install flash-attn --no-build-isolation

# Step 5: Install evo2 and esm
echo "Installing evo2, esm..."
uv pip install evo2 "esm>=3.0"

# Step 6: Install manylatents (for Evo2Encoder base class)
echo "Installing manylatents..."
uv pip install "manylatents @ git+https://github.com/latent-reasoning-works/manylatents.git@main" --index-strategy unsafe-best-match
uv pip install -e . --no-deps  # Install local package

echo "DNA venv ready: $(python -c 'import torch; print(f\"torch {torch.__version__}\")')"
echo "Verifying imports..."
python -c "import evo2; print('evo2 OK'); import esm; print('esm OK')"
python -c "from manylatents.dogma.encoders import Evo2Encoder; print('Evo2Encoder OK')"
