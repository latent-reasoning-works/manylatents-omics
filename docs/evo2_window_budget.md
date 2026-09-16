# Evo 2 teacher window budget

Record for Task 5, ahead of a cluster run. `scripts/cache_evo2_teacher.sbatch`
caches one (model, window) point per submission through
`manylatents/dogma/algorithms/evo2_operator_cache.py`, and appends a JSON line
of peak memory, wall-clock per batch, actual batch sizes and OOM events to
`evo2_window_budget.jsonl` at `RESULTS`. This document is the place those
numbers land; **the sweep has not been run yet** -- this worker does not SSH
or submit jobs (no `sbatch`/`srun`; that's `shop`'s job), so the table below
is the plan and the harness, not a result. The harness itself has been
reviewed and fixed since the previous pass: it now warms the model up before
timing, filters variant IDs and sequences as pairs rather than two lists
filtered independently, and rejects sequences shorter than the requested
window instead of silently measuring a smaller one (see
`tests/dogma/test_evo2_operator_cache.py`).

## The sweep

12 submissions: `{evo2_1b_base, evo2_7b, evo2_40b}` x `{1024, 2048, 4096, 8192}`
bp, e.g.:

```bash
for model in evo2_1b_base evo2_7b; do
  for window in 1024 2048 4096 8192; do
    sbatch --export=MODEL_NAME=$model,WINDOW_BP=$window \
      scripts/cache_evo2_teacher.sbatch
  done
done

# 40B needs an h200 node -- see below.
for window in 1024 2048 4096 8192; do
  sbatch --partition=<h200-partition> --gpus-per-node=h200:1 \
    --export=MODEL_NAME=evo2_40b,WINDOW_BP=$window \
    scripts/cache_evo2_teacher.sbatch
done
```

Each job loads a `ClinVarDataModule` batch of DNA sequences, pairs each ID
with its sequence and centers it on the variant at the requested `window_bp`
(`pair_and_window_sequences` -- sequences shorter than `window_bp` are
excluded and reported rather than silently measured at a smaller window),
loads and warms up `Evo2Encoder` at that model's mid-depth layer
(`blocks.14.mlp.l3` / `blocks.16.mlp.l3` / `blocks.25.mlp.l3` for 1B/7B/40B)
before measuring, and records `torch.cuda.max_memory_allocated`, wall-clock
divided by the number of micro-batches *actually run*, the actual size of
each micro-batch, and any OOM event -- so a completed row states what batch
size it completed at rather than assuming the requested one held.

## Results

| model | window_bp | peak_mem_gb | wall_clock_s_per_batch | actual_batch_sizes | num_oom_events | notes |
|---|---|---|---|---|---|---|
| evo2_1b_base | 1024 | _pending_ | _pending_ | _pending_ | _pending_ | |
| evo2_1b_base | 2048 | _pending_ | _pending_ | _pending_ | _pending_ | |
| evo2_1b_base | 4096 | _pending_ | _pending_ | _pending_ | _pending_ | |
| evo2_1b_base | 8192 | _pending_ | _pending_ | _pending_ | _pending_ | |
| evo2_7b | 1024 | _pending_ | _pending_ | _pending_ | _pending_ | |
| evo2_7b | 2048 | _pending_ | _pending_ | _pending_ | _pending_ | |
| evo2_7b | 4096 | _pending_ | _pending_ | _pending_ | _pending_ | |
| evo2_7b | 8192 | _pending_ | _pending_ | _pending_ | _pending_ | |
| evo2_40b | 1024 | _pending_ | _pending_ | _pending_ | _pending_ | needs h200 |
| evo2_40b | 2048 | _pending_ | _pending_ | _pending_ | _pending_ | needs h200 |
| evo2_40b | 4096 | _pending_ | _pending_ | _pending_ | _pending_ | needs h200 |
| evo2_40b | 8192 | _pending_ | _pending_ | _pending_ | _pending_ | needs h200 |

Fill this table from `evo2_window_budget.jsonl` once the sweep has run, and
replace "_pending_" rather than leaving stale numbers if a job OOMs. The
harness now records the actual per-batch size and OOM count directly (it no
longer relies on `encode_batch`'s own silent halve-and-retry), so a row
completing at all does not mean it completed at the requested `batch_size` --
read `actual_batch_sizes` and `num_oom_events` from the JSON line, not just
`peak_mem_gb`.

## What is already known without running it

82.25 GB of bf16 40B parameters sit on an H200's 141 GB, leaving roughly 59 GB
for activations, KV-equivalent state, and the pooling reduction -- room at a
short context, and, per `docs/evo2_40b_staging.md`, none at all on an 80 GB
H100. The 1B and 7B fit comfortably on one H100 at every window in the sweep;
their rows exist to give the 40B's numbers a baseline, not because their
budget is in doubt.

## Which window later tasks use

**Not yet decided.** This is deliberately left open rather than guessed: the
choice is the largest `window_bp` at which all three teachers complete the
sweep without OOM, and that is exactly the number the table above is missing.
Once the sweep runs, record the chosen window here and have Tasks 6 and 7
read it from `manifest.json`'s `window_bp` rather than from this line, so a
later change to the choice cannot leave the two out of sync.
