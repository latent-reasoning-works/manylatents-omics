#!/bin/bash
# Setup script for Tamia cluster after rsync transfer from Mila
# Run on Tamia login node:
#   bash /scratch/c/cesarmvc/merging-dogma/cross-dogma/experiments/tools/manylatents-omics/scripts/setup-tamia.sh
set -euo pipefail

BASE="/scratch/c/cesarmvc/merging-dogma/cross-dogma/experiments/tools/manylatents-omics"
MODELS="/scratch/c/cesarmvc/merging-dogma/models/huggingface"

echo "=== Tamia Setup ==="
echo ""

# 1. Fix venv Python paths (Mila uses /cvmfs, Tamia uses modules)
echo "[1/4] Loading modules..."
module load python/3.12 cuda/12.6 2>/dev/null

TAMIA_PYTHON=$(which python3)
echo "  Python: $TAMIA_PYTHON"

echo "[2/4] Fixing .venv pyvenv.cfg..."
if [ -f "$BASE/.venv/pyvenv.cfg" ]; then
    # Update home to Tamia's Python directory
    PYTHON_DIR=$(dirname "$TAMIA_PYTHON")
    sed -i "s|^home = .*|home = $PYTHON_DIR|" "$BASE/.venv/pyvenv.cfg"
    echo "  .venv home → $PYTHON_DIR"
fi

if [ -f "$BASE/.venv-dna/pyvenv.cfg" ]; then
    PYTHON_DIR=$(dirname "$TAMIA_PYTHON")
    sed -i "s|^home = .*|home = $PYTHON_DIR|" "$BASE/.venv-dna/pyvenv.cfg"
    echo "  .venv-dna home → $PYTHON_DIR"
fi

# 3. Install manylatents + manylatents-omics in editable mode (offline)
echo "[3/4] Installing packages in editable mode..."
MANYLATENTS_BASE="$BASE/../manylatents"

# .venv
export VIRTUAL_ENV="$BASE/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
pip install --no-deps -e "$MANYLATENTS_BASE" 2>/dev/null && echo "  .venv: manylatents installed" || echo "  .venv: manylatents install failed"
pip install --no-deps -e "$BASE" 2>/dev/null && echo "  .venv: manylatents-omics installed" || echo "  .venv: manylatents-omics install failed"

# .venv-dna
export VIRTUAL_ENV="$BASE/.venv-dna"
export PATH="$VIRTUAL_ENV/bin:$PATH"
pip install --no-deps -e "$MANYLATENTS_BASE" 2>/dev/null && echo "  .venv-dna: manylatents installed" || echo "  .venv-dna: manylatents install failed"
pip install --no-deps -e "$BASE" 2>/dev/null && echo "  .venv-dna: manylatents-omics installed" || echo "  .venv-dna: manylatents-omics install failed"

# 4. Verify
echo "[4/4] Verifying imports..."
"$BASE/.venv/bin/python" -c "
import torch
print(f'  .venv torch: {torch.__version__}')
import esm
print('  .venv esm: OK')
" 2>&1 || echo "  .venv verification failed"

"$BASE/.venv-dna/bin/python" -c "
import torch
print(f'  .venv-dna torch: {torch.__version__}')
" 2>&1 || echo "  .venv-dna verification failed"

echo ""
echo "=== Setup complete ==="
echo ""
echo "HF_HOME should be set to: $MODELS"
echo "Submit jobs from: $BASE"
echo ""
echo "Example submission:"
echo "  cd $BASE"
echo "  python -m manylatents.main -m \\"
echo "    experiment=clinvar_delta_protein \\"
echo "    cluster=tamia_submitit \\"
echo "    resources=gpu_tamia_rna \\"
echo "    data.variant_type=missense \\"
echo "    algorithms.latent.channel=wt,mut \\"
echo "    data.valid_variants_file=$BASE/../../data/clinvar/variants/missense_valid_shared.txt \\"
echo "    data_dir=$BASE/../../data \\"
echo "    output_dir=$BASE/../../results"
