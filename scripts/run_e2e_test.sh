#!/bin/bash
#SBATCH --job-name=clinvar-e2e
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/clinvar_e2e_%j.out
#SBATCH --error=logs/clinvar_e2e_%j.err

# ClinVar E2E Fusion Test
# Encodes DNA+Protein, fuses, logs to WandB

set -e

# Set CUDA paths (module load doesn't work on all nodes)
export CUDA_HOME=/cvmfs/ai.mila.quebec/apps/arch/common/cuda/12.4.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=========================================="
echo "ClinVar E2E Fusion Test - Job $SLURM_JOB_ID"
echo "=========================================="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

cd /network/scratch/c/cesar.valdez/lrw/omics

# Run the test
uv run python scripts/test_clinvar_e2e.py

echo ""
echo "Job completed successfully!"
