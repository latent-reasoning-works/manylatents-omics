#!/usr/bin/env bash
#
# Run Workshop Paper Experiments
#
# This script runs the complete set of experiments for the workshop paper:
# 1. Baseline classifiers (single modality)
# 2. Shared subspace analysis (multi-modal fusion + loadings)
# 3. Deviation analysis (outlier scores vs pathogenicity)
#
# Prerequisites:
#   - Precomputed embeddings at ${OUTPUT_DIR}/embeddings/clinvar/
#   - Files: esm3.pt, evo2.pt, orthrus.pt (optional), labels.pt
#
# Usage:
#   ./scripts/run_workshop_experiments.sh           # Local execution
#   ./scripts/run_workshop_experiments.sh --cluster # Cluster submission

set -e

# Configuration
OUTPUT_DIR="${MANYLATENTS_OUTPUT_DIR:-outputs}"
CLUSTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="cluster=mila_submitit resources=gpu"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Workshop Paper Experiments"
echo "=============================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Cluster mode: ${CLUSTER:-local}"
echo ""

# ============================================
# 1. Baseline Classifiers (Single Modality)
# ============================================
echo "=== 1. Baseline Classifiers ==="

echo "1a. Protein-only baseline (ESM3)..."
python -m manylatents.main --config-name=config \
    experiment=clinvar/baselines \
    data.channels=[esm3] \
    name=clinvar_baseline_protein \
    ${CLUSTER}

echo "1b. DNA-only baseline (Evo2)..."
python -m manylatents.main --config-name=config \
    experiment=clinvar/baselines \
    data.channels=[evo2] \
    name=clinvar_baseline_dna \
    ${CLUSTER}

# Optional: RNA baseline (if orthrus.pt exists)
if [[ -f "${OUTPUT_DIR}/embeddings/clinvar/orthrus.pt" ]]; then
    echo "1c. RNA-only baseline (Orthrus)..."
    python -m manylatents.main --config-name=config \
        experiment=clinvar/baselines \
        data.channels=[orthrus] \
        name=clinvar_baseline_rna \
        ${CLUSTER}
fi

echo "1d. Multi-modal baseline (concatenated)..."
python -m manylatents.main --config-name=config \
    experiment=clinvar/baselines \
    data.channels=[esm3,evo2] \
    name=clinvar_baseline_concat \
    ${CLUSTER}

# ============================================
# 2. Shared Subspace Analysis
# ============================================
echo ""
echo "=== 2. Shared Subspace Analysis ==="

echo "2a. DNA + Protein shared subspace..."
python -m manylatents.main --config-name=config \
    experiment=clinvar/shared_subspace \
    data.channels=[esm3,evo2] \
    'callbacks.embedding.loadings_analysis.modality_dims=[1536,1920]' \
    'callbacks.embedding.loadings_analysis.modality_names=[protein,dna]' \
    name=clinvar_shared_dna_protein \
    ${CLUSTER}

# Optional: Include RNA if available
if [[ -f "${OUTPUT_DIR}/embeddings/clinvar/orthrus.pt" ]]; then
    echo "2b. DNA + RNA + Protein shared subspace..."
    python -m manylatents.main --config-name=config \
        experiment=clinvar/shared_subspace \
        data.channels=[esm3,evo2,orthrus] \
        'callbacks.embedding.loadings_analysis.modality_dims=[1536,1920,256]' \
        'callbacks.embedding.loadings_analysis.modality_names=[protein,dna,rna]' \
        name=clinvar_shared_all \
        ${CLUSTER}
fi

# ============================================
# 3. Deviation Analysis
# ============================================
echo ""
echo "=== 3. Deviation Analysis ==="

echo "3a. Deviation analysis (k=20)..."
python -m manylatents.main --config-name=config \
    experiment=clinvar/deviation \
    data.channels=[esm3,evo2] \
    metrics.embedding.outlier_score.k=20 \
    name=clinvar_deviation_k20 \
    ${CLUSTER}

echo "3b. Deviation analysis (k=50)..."
python -m manylatents.main --config-name=config \
    experiment=clinvar/deviation \
    data.channels=[esm3,evo2] \
    metrics.embedding.outlier_score.k=50 \
    name=clinvar_deviation_k50 \
    ${CLUSTER}

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "Experiments Complete"
echo "=============================================="
echo ""
echo "Results logged to WandB project: merging-dogma"
echo ""
echo "Key metrics to check:"
echo "  - Baselines: embedding.auc"
echo "  - Shared Subspace: loadings/n_shared, loadings/shared_fraction"
echo "  - Deviation: outlier_score/auc, outlier_score/mean"
