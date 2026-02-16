#!/bin/bash
#SBATCH --job-name=lrw-smoke-test
#SBATCH --account=aip-wolfg
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=smoke_test_%j.out
#
# Smoke test: validates the full manylatents-omics chain on a compute node.
# Runs PCA on synthetic swissroll, logs embedding plot to WandB offline.
#
# Usage:
#   sbatch scripts/smoke-test.sbatch
#   # or via deploy script:
#   bash scripts/deploy.sh tamia --smoke-test

set -e

echo "=== LRW Smoke Test ==="
echo "Host: $(hostname)"
echo "Job:  $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# Environment
module load cuda/12.6 python/3.12
cd "$SLURM_SUBMIT_DIR"
export VIRTUAL_ENV=$PWD/.venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

# 1. Basic imports
echo "[1/4] Import check..."
python -c "
import torch
print(f'  torch {torch.__version__}, CUDA {torch.cuda.is_available()}')
import manylatents
print(f'  manylatents namespace: {manylatents.__path__}')
from manylatents.omics_plugin import OmicsSearchPathPlugin
print('  OmicsSearchPathPlugin OK')
from manylatents.configs import Config
print('  ManylatentsSearchPathPlugin OK')
"

# 2. Plugin chain
echo "[2/4] Plugin registration..."
python -c "
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from manylatents.omics_plugin import OmicsSearchPathPlugin

plugins = Plugins.instance()
plugins.register(OmicsSearchPathPlugin)
discovered = [p.__name__ for p in plugins.discover(SearchPathPlugin)]
print(f'  Registered plugins: {discovered}')
assert 'OmicsSearchPathPlugin' in discovered
print('  Plugin chain OK')
"

# 3. Run PCA on swissroll with embedding plot
echo "[3/4] Running PCA on swissroll..."
python -m manylatents.omics.main \
    --config-name=config \
    experiment=single_algorithm \
    data=swissroll \
    algorithms/latent=pca \
    algorithms.latent.n_components=2 \
    callbacks/embedding=default \
    project=lrw-smoke-test \
    name="smoke_$(hostname)_${SLURM_JOB_ID}"

# 4. Verify outputs
echo "[4/4] Verifying outputs..."
WANDB_DIR=$(ls -td wandb/run-* 2>/dev/null | head -1)
if [ -n "$WANDB_DIR" ]; then
    echo "  WandB run: $WANDB_DIR"
    ls "$WANDB_DIR/files/" 2>/dev/null | head -5
    echo "  WandB offline logs OK"
else
    echo "  WARNING: No WandB run directory found"
fi

# Check for embedding plot
PLOT=$(find logs/ -name '*.png' -newer /proc/$$/fd/0 2>/dev/null | head -1)
if [ -n "$PLOT" ]; then
    echo "  Plot: $PLOT"
else
    echo "  WARNING: No embedding plot found"
fi

echo ""
echo "=== Smoke test PASSED ==="
echo "Sync WandB: wandb sync $WANDB_DIR"
