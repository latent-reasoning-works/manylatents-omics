#!/bin/bash
# Setup DNA encoder venv (Evo2 + ESM3) - UNLOCKED
# torch 2.8+cu126 via uv pip (bypasses lock file conflict)
#
# MUST RUN ON GPU NODE with cudnn (for transformer-engine build)
# Usage: sbatch -p gpu --gres=gpu:1 --wrap="source scripts/setup-dna-venv.sh"
#    or: salloc -p gpu --gres=gpu:1 then source scripts/setup-dna-venv.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

module load cuda/12.4.1 2>/dev/null || true
module load cuda/12.4.1/cudnn/9.3 2>/dev/null || true

# Create venv if needed
if [ ! -d .venv-dna ]; then
    echo "Creating DNA venv..."
    uv venv .venv-dna --python 3.12
fi

# Install core packages
echo "Installing evo2, esm, torch..."
VIRTUAL_ENV="$PROJECT_DIR/.venv-dna" uv pip install \
    "esm>=3.0" "evo2" "torch>=2.8" \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple

# Install transformer-engine with pytorch bindings (needs cudnn)
echo "Installing transformer-engine[pytorch]..."
VIRTUAL_ENV="$PROJECT_DIR/.venv-dna" uv pip install \
    "transformer-engine[pytorch]" \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple

export VIRTUAL_ENV="$PROJECT_DIR/.venv-dna"
export PATH="$VIRTUAL_ENV/bin:$PATH"
echo "DNA venv ready: $(python -c 'import torch; print(f\"torch {torch.__version__}\")')"
