#!/bin/bash
# Setup RNA encoder venv (Orthrus + ESM3) with torch 2.5.1
# Uses main .venv directory with mamba-ssm prebuilt wheels
#
# Usage: source scripts/setup-rna-venv.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load CUDA for mamba-ssm
module load cuda/12.1.1 2>/dev/null || true

# Create/update RNA venv
if [ ! -d .venv ] || [ ! -f .venv/pyvenv.cfg ]; then
    echo "Creating RNA venv with torch 2.5.1..."
    uv sync --extra dogma-rna --index-strategy unsafe-best-match
    echo "RNA venv created at .venv"
else
    echo "RNA venv exists at .venv"
fi

# Activate RNA venv
export VIRTUAL_ENV="$PROJECT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
echo "Activated RNA venv: $VIRTUAL_ENV"
