#!/bin/bash
#SBATCH --job-name=spike_evo2_trainable
#SBATCH --account=aip-wolfg
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/c/cesarmvc/merging-dogma/logs/spike_evo2_trainable_%j.out
#SBATCH --error=/scratch/c/cesarmvc/merging-dogma/logs/spike_evo2_trainable_%j.err
set -uo pipefail
module --ignore_cache load cuda/12.6 python/3.12

PROJECT=/scratch/c/cesarmvc/merging-dogma
OMICS=$PROJECT/cross-dogma/experiments/tools/manylatents-omics
export HF_HOME=$PROJECT/models/huggingface
export EVO2_1B_PT=$(find $HF_HOME/hub/models--arcinstitute--evo2_1b_base -name "evo2_1b_base.pt" | head -1)
export CUDA_VISIBLE_DEVICES=0
echo "checkpoint=$EVO2_1B_PT"

"$OMICS/.venv-dna/bin/python" -u "$PROJECT/spike_evo2_trainable.py"
# Capture before the trailing echo: it would reset $? and SLURM would record
# COMPLETED/0:0 for a spike that died.
status=$?
echo "spike exit=$status"
exit $status
