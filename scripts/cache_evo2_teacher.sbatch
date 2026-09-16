#!/bin/bash
#SBATCH --job-name=cache_evo2_teacher
#SBATCH --account=aip-wolfg
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/c/cesarmvc/merging-dogma/logs/cache_evo2_teacher_%j.out
#SBATCH --error=/scratch/c/cesarmvc/merging-dogma/logs/cache_evo2_teacher_%j.err
#
# Caches ONE frozen Evo2 teacher's pooled ClinVar representations plus the
# sigma its operator is built from (manylatents/dogma/algorithms/
# evo2_operator_cache.py), and records peak GPU memory and wall-clock per
# batch for the window budget in docs/evo2_window_budget.md. Measurement
# starts only after the model is loaded and warmed up, and records the
# actual per-batch size and any OOM event rather than assuming the
# requested batch size held; sequences shorter than WINDOW_BP are excluded
# and reported rather than silently measured at a smaller window.
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline

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
from pathlib import Path

import torch

from manylatents.dogma.algorithms.evo2_operator_cache import (
    measure_teacher,
    pair_and_window_sequences,
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

dm = ClinVarDataModule(data_dir=clinvar_dir, max_variants=max_variants, batch_size=batch_size)
dm.setup()
# IDs and sequences are filtered and windowed together here, never as two
# independently-filtered lists: an ID list that drops a missing sequence by
# position rather than by pairing shifts every later ID out of alignment
# with the representation it is meant to describe.
variant_ids, sequences, insufficient = pair_and_window_sequences(
    dm.get_variant_ids(), dm.get_sequences()["dna"], window_bp,
)
if not sequences:
    raise ValueError(
        f"no sequence has >= {window_bp}bp of context; "
        f"longest available is {max((n for _, n in insufficient), default=0)}bp"
    )

encoder = Evo2Encoder(
    model_name=model_name,
    layer_name=layer,
    weights_path=weights_path or None,
    device="cuda",
)

# Model loading is excluded from batch timing but terminal load OOMs still
# produce a results row. Warmup and measured batches share explicit backoff.
try:
    encoder._ensure_loaded()
except torch.cuda.OutOfMemoryError:
    representations = None
    measurement = {
        "status": "oom", "failure_phase": "model_load", "num_oom_events": 1,
        "oom_phases": ["model_load"], "actual_batch_sizes": [], "num_batches": 0,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        "wall_clock_s_per_batch": None,
    }
else:
    representations, measurement = measure_teacher(encoder, sequences, batch_size)

cache_dir = Path(cache_root) / model_name / f"window_{window_bp}"
if representations is not None:
    try:
        sigma = teacher_sigma(representations, quantile=0.25)
        write_teacher_cache(
            cache_dir, representations, sigma=sigma, layer=layer,
            model_name=model_name, window_bp=window_bp, pooling="masked_mean",
            variant_ids=variant_ids,
        )
    except Exception as exc:
        measurement.update(status="cache_error", error=str(exc))

record = {
    "model_name": model_name,
    "window_bp": window_bp,
    "layer": layer,
    "num_sequences": len(sequences),
    "num_insufficient_context": len(insufficient),
    "insufficient_context_lengths": sorted({n for _, n in insufficient}),
    "requested_batch_size": batch_size,
    **measurement,
    "cache_dir": str(cache_dir) if measurement["status"] == "completed" else None,
}
print(json.dumps(record))

results_file = Path(results_path)
results_file.parent.mkdir(parents=True, exist_ok=True)
with open(results_file, "a") as f:
    f.write(json.dumps(record) + "\n")
if record["status"] != "completed":
    sys.exit(1)
PYEOF
# Capture before the trailing echo: it would reset $? and SLURM would record
# COMPLETED/0:0 for a run that died.
status=$?
echo "cache_evo2_teacher exit=$status"
exit $status
