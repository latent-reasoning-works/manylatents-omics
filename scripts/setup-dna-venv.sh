#!/bin/bash
# Setup DNA encoder venv (Evo2 + ESM3) with torch 2.8+
# Creates .venv-dna directory with transformer-engine prebuilt wheels
#
# Usage: source scripts/setup-dna-venv.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load CUDA for transformer-engine
module load cuda/12.4.1 2>/dev/null || true

# Create DNA venv if needed
if [ ! -d .venv-dna ]; then
    echo "Creating DNA venv with torch 2.8+..."

    # Temporarily swap pyproject.toml
    cp pyproject.toml pyproject-rna.toml.bak
    cp pyproject-dna.toml pyproject.toml

    # Sync DNA venv
    UV_PROJECT_ENVIRONMENT=.venv-dna uv lock --index-strategy unsafe-best-match
    UV_PROJECT_ENVIRONMENT=.venv-dna uv sync --extra dogma-dna --index-strategy unsafe-best-match

    # Restore RNA pyproject.toml
    mv pyproject-rna.toml.bak pyproject.toml

    echo "DNA venv created at .venv-dna"
fi

# Activate DNA venv
export VIRTUAL_ENV="$PROJECT_DIR/.venv-dna"
export PATH="$VIRTUAL_ENV/bin:$PATH"
echo "Activated DNA venv: $VIRTUAL_ENV"
