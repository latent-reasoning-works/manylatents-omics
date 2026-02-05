#!/bin/bash
# Setup RNA encoder venv (Orthrus + ESM3) - LOCKED
# torch 2.5.1+cu121 with mamba-ssm prebuilt wheels
#
# Usage: source scripts/setup-rna-venv.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

module load cuda/12.1.1 2>/dev/null || true

# Standard uv sync with lock file
uv sync --extra dogma --index-strategy unsafe-best-match

export VIRTUAL_ENV="$PROJECT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
echo "RNA venv ready: $(python -c 'import torch; print(f\"torch {torch.__version__}\")')"
