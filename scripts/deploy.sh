#!/bin/bash
# Deploy manylatents + omics to an offline cluster
#
# Syncs code and venvs from Mila, fixes Python paths, runs smoke test.
# Must be run from Mila (internet access required for initial venv build).
#
# Usage:
#   bash scripts/deploy.sh tamia                    # deploy to tamia
#   bash scripts/deploy.sh tamia --skip-venv        # code only, keep existing venvs
#   bash scripts/deploy.sh tamia --smoke-test       # deploy + submit smoke test
#
# Prerequisites:
#   - SSH config for target cluster (e.g., Host tamia in ~/.ssh/config)
#   - Venvs built locally: uv sync --extra dogma && source scripts/setup-dna-venv.sh

set -euo pipefail

CLUSTER="${1:?Usage: deploy.sh <cluster> [--skip-venv] [--smoke-test]}"
shift

SKIP_VENV=false
SMOKE_TEST=false
for arg in "$@"; do
    case $arg in
        --skip-venv) SKIP_VENV=true ;;
        --smoke-test) SMOKE_TEST=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# Paths — override with env vars if needed
OMICS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANYLATENTS_DIR="${MANYLATENTS_DIR:-$(cd "$OMICS_DIR/../manylatents" && pwd)}"

# Remote paths per cluster
case "$CLUSTER" in
    tamia)
        REMOTE_BASE="/scratch/c/cesarmvc/lrw"
        SETUP_MODULES="cuda/12.6 python/3.12"
        ;;
    narval|cedar|beluga)
        REMOTE_BASE="/scratch/cesarmvc/lrw"
        SETUP_MODULES="cuda/12.6 python/3.12"
        ;;
    *)
        echo "Unknown cluster: $CLUSTER"
        echo "Add remote paths for this cluster in deploy.sh"
        exit 1
        ;;
esac

REMOTE_OMICS="$REMOTE_BASE/omics"
REMOTE_ML="$REMOTE_BASE/manylatents"

echo "=== Deploying to $CLUSTER ==="
echo "  Local omics:       $OMICS_DIR"
echo "  Local manylatents: $MANYLATENTS_DIR"
echo "  Remote:            $REMOTE_BASE"
echo ""

# 1. Create remote directories
echo "[1/4] Creating remote directories..."
ssh "$CLUSTER" "mkdir -p $REMOTE_OMICS $REMOTE_ML"

# 2. Sync code
echo "[2/4] Syncing code..."
rsync -avz --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'wandb/' \
    --exclude 'logs/' \
    --exclude 'outputs/' \
    --exclude '*.out' \
    --exclude '.venv*' \
    "$MANYLATENTS_DIR/" "$CLUSTER:$REMOTE_ML/"

rsync -avz --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'wandb/' \
    --exclude 'logs/' \
    --exclude 'outputs/' \
    --exclude '*.out' \
    --exclude '.venv*' \
    "$OMICS_DIR/" "$CLUSTER:$REMOTE_OMICS/"

# 3. Sync venvs (optional)
if [ "$SKIP_VENV" = false ]; then
    echo "[3/4] Syncing venvs..."

    # Check local venvs exist
    if [ ! -d "$OMICS_DIR/.venv" ]; then
        echo "  ERROR: No .venv found. Run 'uv sync --extra dogma' first."
        exit 1
    fi

    rsync -avz --delete \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        "$OMICS_DIR/.venv/" "$CLUSTER:$REMOTE_OMICS/.venv/"

    if [ -d "$OMICS_DIR/.venv-dna" ]; then
        rsync -avz --delete \
            --exclude '__pycache__' \
            --exclude '*.pyc' \
            "$OMICS_DIR/.venv-dna/" "$CLUSTER:$REMOTE_OMICS/.venv-dna/"
    fi

    # Fix Python paths in venvs
    echo "  Fixing venv Python paths..."
    ssh "$CLUSTER" "
        module load $SETUP_MODULES 2>/dev/null
        PYTHON_DIR=\$(dirname \$(which python3))
        for cfg in $REMOTE_OMICS/.venv/pyvenv.cfg $REMOTE_OMICS/.venv-dna/pyvenv.cfg; do
            if [ -f \"\$cfg\" ]; then
                sed -i \"s|^home = .*|home = \$PYTHON_DIR|\" \"\$cfg\"
                echo \"  Fixed: \$cfg\"
            fi
        done
    "

    # Install packages in editable mode (no network needed)
    echo "  Installing packages in editable mode..."
    ssh "$CLUSTER" "
        module load $SETUP_MODULES 2>/dev/null
        export VIRTUAL_ENV=$REMOTE_OMICS/.venv
        export PATH=\$VIRTUAL_ENV/bin:\$PATH
        pip install --no-deps -e $REMOTE_ML 2>/dev/null && echo '  .venv: manylatents OK'
        pip install --no-deps -e $REMOTE_OMICS 2>/dev/null && echo '  .venv: omics OK'
    "
else
    echo "[3/4] Skipping venv sync (--skip-venv)"
fi

# 4. Smoke test (optional)
if [ "$SMOKE_TEST" = true ]; then
    echo "[4/4] Submitting smoke test..."
    rsync -avz "$OMICS_DIR/scripts/smoke-test.sbatch" "$CLUSTER:$REMOTE_OMICS/scripts/"
    ssh "$CLUSTER" "cd $REMOTE_OMICS && sbatch scripts/smoke-test.sbatch"
else
    echo "[4/4] Skipping smoke test (use --smoke-test to enable)"
fi

echo ""
echo "=== Deploy complete ==="
echo "  Code:  $CLUSTER:$REMOTE_BASE"
echo "  Venvs: $([ "$SKIP_VENV" = true ] && echo 'unchanged' || echo 'synced')"
echo ""
echo "To run manually:"
echo "  ssh $CLUSTER"
echo "  cd $REMOTE_OMICS"
echo "  sbatch scripts/smoke-test.sbatch"
