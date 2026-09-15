#!/bin/bash
#SBATCH --job-name=cache_evo2_teacher
#SBATCH --account=aip-wolfg
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/c/cesarmvc/merging-dogma/logs/cache_evo2_teacher_%j.out
#SBATCH --error=/scratch/c/cesarmvc/merging-dogma/logs/cache_evo2_teacher_%j.err
#
# Caches ONE frozen Evo2 teacher's pooled ClinVar representations plus the
# sigma its operator is built from (manylatents/dogma/algorithms/
# evo2_operator_cache.py), and records peak GPU memory and wall-clock per
# batch for the window budget in docs/evo2_window_budget.md.
#
# MODEL_NAME and WINDOW_BP select one point of the sweep; submit once per
# (model, window) pair rather than hardcoding a matrix here, e.g.:
#
#   for model in evo2_1b_base evo2_7b; do
#     for window in 1024 2048 4096 8192; do
#       sbatch --export=MODEL_NAME=$model,WINDOW_BP=$window \
#         scripts/cache_evo2_teacher.sbatch
#     done
#   done
#
# The 40B needs an h200 node -- 82.25 GB of bf16 parameters on an 80 GB H100
# leaves no room at all. Override the partition and GPU type for it, never
# hardcode a node list:
#
#   sbatch --partition=<h200-partition> --gpus-per-node=h200:1 \
#     --export=MODEL_NAME=evo2_40b,WINDOW_BP=1024 \
#     scripts/cache_evo2_teacher.sbatch
set -uo pipefail
module --ignore_cache load cuda/12.6 python/3.12

PROJECT=/scratch/c/cesarmvc/merging-dogma
OMICS=$PROJECT/cross-dogma/experiments/tools/manylatents-omics
export HF_HOME=$PROJECT/models/huggingface
# 40B is a flat two-part .pt staged by scripts/stage_evo2_40b.sh and resolved
# by Evo2Encoder through this variable; 1B/7B live in the HF hub cache layout
# below and are located explicitly instead.
export EVO2_WEIGHTS_DIR=$PROJECT/models
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to evo2_1b_base, evo2_7b or evo2_40b}"
WINDOW_BP="${WINDOW_BP:?set WINDOW_BP, e.g. 1024, 2048, 4096 or 8192}"
CLINVAR_DIR="${CLINVAR_DIR:-$PROJECT/data/clinvar}"
MAX_VARIANTS="${MAX_VARIANTS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT/caches/evo2_teacher}"
RESULTS="${RESULTS:-$PROJECT/logs/evo2_window_budget.jsonl}"

# Mid-depth layers only, never penultimate and never final. Upstream depths
# are 25 (1B), 32 (7B), 50 (40B); see manylatents/dogma/encoders/evo2.py.
case "$MODEL_NAME" in
  evo2_1b_base) LAYER="blocks.14.mlp.l3" ;;
  evo2_7b)      LAYER="blocks.16.mlp.l3" ;;
  evo2_40b)     LAYER="blocks.25.mlp.l3" ;;
  *) echo "unknown MODEL_NAME=$MODEL_NAME" >&2; exit 1 ;;
esac

if [[ "$MODEL_NAME" == "evo2_40b" ]]; then
  WEIGHTS_PATH=""
else
  WEIGHTS_PATH=$(find "$HF_HOME/hub/models--arcinstitute--${MODEL_NAME}" \
    -name "${MODEL_NAME}.pt" 2>/dev/null | head -1)
fi

echo "model=$MODEL_NAME window_bp=$WINDOW_BP layer=$LAYER weights_path=${WEIGHTS_PATH:-<EVO2_WEIGHTS_DIR>}"

# Run the committed harness from its repository path: a copy dropped beside
# the project root cannot be reproduced from a checkout.
cd "$OMICS"
"$OMICS/.venv-dna/bin/python3" -u - \
  "$MODEL_NAME" "$WINDOW_BP" "$LAYER" "$WEIGHTS_PATH" "$CLINVAR_DIR" \
  "$MAX_VARIANTS" "$BATCH_SIZE" "$CACHE_ROOT" "$RESULTS" << 'PYEOF'
import json
import sys
import time
from pathlib import Path

import torch

from manylatents.dogma.algorithms.evo2_operator_cache import (
    teacher_sigma,
    write_teacher_cache,
)
from manylatents.dogma.data import ClinVarDataModule
from manylatents.dogma.encoders.evo2 import Evo2Encoder

(model_name, window_bp, layer, weights_path, clinvar_dir,
 max_variants, batch_size, cache_root, results_path) = sys.argv[1:10]
window_bp = int(window_bp)
max_variants = int(max_variants)
batch_size = int(batch_size)


def center_window(seq: str, window_bp: int) -> str:
    """The ClinVar FASTA is centered on the variant; take window_bp around
    that same center rather than the sequence start, so a shorter window
    still covers the variant."""
    if len(seq) <= window_bp:
        return seq
    mid = len(seq) // 2
    half = window_bp // 2
    return seq[mid - half : mid - half + window_bp]


dm = ClinVarDataModule(data_dir=clinvar_dir, max_variants=max_variants, batch_size=batch_size)
dm.setup()
variant_ids = dm.get_variant_ids()
sequences = [center_window(s, window_bp) for s in dm.get_sequences()["dna"] if s]

encoder = Evo2Encoder(
    model_name=model_name,
    layer_name=layer,
    weights_path=weights_path or None,
    device="cuda",
)

torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
representations = encoder.encode_batch(sequences, batch_size=batch_size, show_progress=False)
torch.cuda.synchronize()
elapsed_s = time.perf_counter() - start
num_batches = max(1, -(-len(sequences) // batch_size))  # ceil division
peak_mem_gb = torch.cuda.max_memory_allocated() / 2**30

sigma = teacher_sigma(representations, quantile=0.25)

cache_dir = Path(cache_root) / model_name / f"window_{window_bp}"
write_teacher_cache(
    cache_dir, representations, sigma=sigma, layer=layer,
    model_name=model_name, window_bp=window_bp, pooling="masked_mean",
    variant_ids=variant_ids[: len(sequences)],
)

record = {
    "model_name": model_name,
    "window_bp": window_bp,
    "layer": layer,
    "num_sequences": len(sequences),
    "batch_size": batch_size,
    "peak_mem_gb": round(peak_mem_gb, 3),
    "wall_clock_s_per_batch": round(elapsed_s / num_batches, 3),
    "cache_dir": str(cache_dir),
}
print(json.dumps(record))

results_file = Path(results_path)
results_file.parent.mkdir(parents=True, exist_ok=True)
with open(results_file, "a") as f:
    f.write(json.dumps(record) + "\n")
PYEOF
# Capture before the trailing echo: it would reset $? and SLURM would record
# COMPLETED/0:0 for a run that died.
status=$?
echo "cache_evo2_teacher exit=$status"
exit $status
