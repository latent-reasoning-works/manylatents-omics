#!/bin/bash
#
# ClinVar Central Dogma Analysis Pipeline
#
# Orchestrates the full pipeline:
#   1. Submit Evo2, Orthrus, ESM3 embedding jobs in parallel
#   2. After all complete, consolidate embeddings
#   3. Compute geometric metrics
#   4. Run fusion evaluation
#
# Usage:
#   ./scripts/run_clinvar_pipeline.sh
#
# Prerequisites:
#   - data/clinvar/dna.fasta
#   - data/clinvar/rna.fasta
#   - data/clinvar/protein.fasta
#   - data/clinvar/variants.tsv
#   - tools/orthrus-embed/.venv (run `uv sync` in that directory first)

set -e

# Configuration
DATA_DIR="data/clinvar"
EMB_DIR="/network/scratch/c/cesar.valdez/embeddings"
RESULTS_DIR="results/clinvar"

echo "=== ClinVar Central Dogma Analysis Pipeline ==="
echo "Data: $DATA_DIR"
echo "Embeddings: $EMB_DIR"
echo "Results: $RESULTS_DIR"
echo ""

# Validate inputs
for f in "$DATA_DIR/dna.fasta" "$DATA_DIR/rna.fasta" "$DATA_DIR/protein.fasta" "$DATA_DIR/variants.tsv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing required file: $f"
        exit 1
    fi
done

# Create directories
mkdir -p logs "$RESULTS_DIR"

echo "=== Stage 2: Submitting Embedding Jobs ==="

# Submit Evo2 (DNA)
JOB_EVO2=$(sbatch --parsable scripts/embed_evo2.sh \
    "$DATA_DIR/dna.fasta" \
    "$EMB_DIR/evo2/clinvar/")
echo "Evo2 job: $JOB_EVO2"

# Submit ESM3 (Protein)
JOB_ESM3=$(sbatch --parsable scripts/embed_esm3.sh \
    "$DATA_DIR/protein.fasta" \
    "$EMB_DIR/esm3/clinvar/")
echo "ESM3 job: $JOB_ESM3"

# Submit Orthrus (RNA) - uses isolated environment
JOB_ORTH=$(sbatch --parsable scripts/embed_orthrus.sh \
    "$DATA_DIR/rna.fasta" \
    "$EMB_DIR/orthrus/clinvar/")
echo "Orthrus job: $JOB_ORTH"

echo ""
echo "=== Stage 3: Submitting Consolidation Job ==="

# Consolidation (after all embedding jobs complete)
JOB_CONS=$(sbatch --parsable \
    --dependency=afterok:$JOB_EVO2:$JOB_ESM3:$JOB_ORTH \
    --job-name=consolidate \
    --output=logs/consolidate-%j.out \
    --time=01:00:00 \
    --mem=32G \
    --cpus-per-task=4 \
    --wrap="cd /network/scratch/c/cesar.valdez/lrw/omics && \
        source .venv/bin/activate && \
        python scripts/consolidate_embeddings.py \
            --variants $DATA_DIR/variants.tsv \
            --evo2-dir $EMB_DIR/evo2/clinvar/ \
            --orthrus-dir $EMB_DIR/orthrus/clinvar/ \
            --esm3-dir $EMB_DIR/esm3/clinvar/ \
            --output $EMB_DIR/clinvar_consolidated.h5")
echo "Consolidation job: $JOB_CONS"

echo ""
echo "=== Stage 4: Submitting Geometric Analysis Job ==="

# Geometric metrics (after consolidation)
JOB_GEO=$(sbatch --parsable \
    --dependency=afterok:$JOB_CONS \
    --job-name=geometric \
    --output=logs/geometric-%j.out \
    --time=01:00:00 \
    --mem=32G \
    --cpus-per-task=4 \
    --wrap="cd /network/scratch/c/cesar.valdez/lrw/omics && \
        source .venv/bin/activate && \
        python scripts/compute_geometric_metrics.py \
            --embeddings $EMB_DIR/clinvar_consolidated.h5 \
            --output $RESULTS_DIR/geometric_metrics.csv \
            --metrics pr,lid,tsa \
            --stratify-by label")
echo "Geometric analysis job: $JOB_GEO"

# Cross-modal agreement (after consolidation)
JOB_AGREE=$(sbatch --parsable \
    --dependency=afterok:$JOB_CONS \
    --job-name=agreement \
    --output=logs/agreement-%j.out \
    --time=01:00:00 \
    --mem=32G \
    --cpus-per-task=4 \
    --wrap="cd /network/scratch/c/cesar.valdez/lrw/omics && \
        source .venv/bin/activate && \
        python scripts/compute_geometric_metrics.py \
            --embeddings $EMB_DIR/clinvar_consolidated.h5 \
            --output $RESULTS_DIR/cross_modal_agreement.csv \
            --metrics pr,lid,tsa \
            --compute-agreement")
echo "Cross-modal agreement job: $JOB_AGREE"

echo ""
echo "=== Stage 5: Submitting Fusion Evaluation Job ==="

# Fusion evaluation (after consolidation)
JOB_FUSION=$(sbatch --parsable \
    --dependency=afterok:$JOB_CONS \
    --job-name=fusion-eval \
    --output=logs/fusion-eval-%j.out \
    --time=02:00:00 \
    --mem=32G \
    --cpus-per-task=8 \
    --wrap="cd /network/scratch/c/cesar.valdez/lrw/omics && \
        source .venv/bin/activate && \
        python scripts/evaluate_fusion.py \
            --embeddings $EMB_DIR/clinvar_consolidated.h5 \
            --output $RESULTS_DIR/fusion_comparison.csv \
            --fusion-strategies concat,concat_norm \
            --classifiers logistic,mlp \
            --cv-folds 5 \
            --single-modality-baselines")
echo "Fusion evaluation job: $JOB_FUSION"

echo ""
echo "=== Pipeline Submitted ==="
echo ""
echo "Job Dependencies:"
echo "  Embedding:     $JOB_EVO2, $JOB_ESM3, $JOB_ORTH (parallel)"
echo "  Consolidation: $JOB_CONS (after embedding)"
echo "  Geometric:     $JOB_GEO (after consolidation)"
echo "  Agreement:     $JOB_AGREE (after consolidation)"
echo "  Fusion:        $JOB_FUSION (after consolidation)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/*.out"
echo ""
echo "Expected outputs:"
echo "  $EMB_DIR/clinvar_consolidated.h5"
echo "  $RESULTS_DIR/geometric_metrics.csv"
echo "  $RESULTS_DIR/cross_modal_agreement.csv"
echo "  $RESULTS_DIR/fusion_comparison.csv"
