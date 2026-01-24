# Change: Add ClinVar Geometric Pipeline

## Status: COMPLETE ✅

All tasks complete. E2E validated on L40S GPU.

**Core PR:** https://github.com/latent-reasoning-works/manylatents/pull/193 (merged)
**E2E WandB:** https://wandb.ai/cesar-valdez-mcgill-university/clinvar-geometric/runs/bqsio66v

## Why

The current ClinVar pipeline uses standalone scripts (`embed_evo2.py`, `embed_esm3.py`, `consolidate_embeddings.py`, `compute_geometric_metrics.py`) that bypass Hydra composition patterns. This violates the manylatents/shop integration model where:
- Experiments should be defined via Hydra configs, not hardcoded scripts
- Cluster submission should use Shop launchers, not custom sbatch wrappers
- Algorithms should be LatentModules, not callbacks or scripts

## What Changes

### New Capabilities

**Data Layer:**
- **ClinVarDataModule**: DataModule for loading preprocessed ClinVar variant sequences
- **PrecomputedEmbeddingsDataModule**: DataModule for loading HDF5 with multiple embedding channels

**Algorithm Layer:**
- **BatchEncoder**: LatentModule wrapping foundation encoders for batch sequence processing
- **PrecomputedFusionModule**: LatentModule that loads cached embeddings and fuses them (concat, weighted, attention)

**Metrics:**
- Use existing manylatents metric registry (PR, LID, TSA) via experiment config
- Sweep over dimensionality reductions (n_components: 5, 50, 100)

### New Configs
- `configs/data/clinvar.yaml` - ClinVar sequence data loading
- `configs/data/precomputed_embeddings.yaml` - Multi-channel HDF5 embeddings
- `configs/experiment/clinvar/encode_dna.yaml` - Evo2 encoding
- `configs/experiment/clinvar/encode_protein.yaml` - ESM3 encoding
- `configs/experiment/clinvar/geometric_analysis.yaml` - Fusion + metrics sweep

### New Scripts
- `scripts/download_clinvar.py` - One-time ClinVar bulk download/preprocessing

### Removed (replaced by Hydra)
- `scripts/embed_evo2.py` - Replaced by `+experiment=clinvar/encode_dna`
- `scripts/embed_esm3.py` - Replaced by `+experiment=clinvar/encode_protein`
- `scripts/embed_evo2.sh` - Replaced by Shop cluster launcher
- `scripts/embed_esm3.sh` - Replaced by Shop cluster launcher
- `scripts/consolidate_embeddings.py` - Replaced by `PrecomputedFusionModule`
- `scripts/compute_geometric_metrics.py` - Replaced by manylatents metrics

## Impact

- Affected specs: clinvar-pipeline (new capability)
- Affected code:
  - `manylatents/dogma/data/clinvar_dataset.py` (new)
  - `manylatents/dogma/data/precomputed_embeddings.py` (new)
  - `manylatents/dogma/algorithms/batch_encoder.py` (new)
  - `manylatents/dogma/algorithms/precomputed_fusion.py` (new)
  - `configs/data/clinvar.yaml` (new)
  - `configs/data/precomputed_embeddings.yaml` (new)
  - `configs/experiment/clinvar/*.yaml` (new)
  - `scripts/download_clinvar.py` (new)

## Design Decisions

### 1. Preprocessed vs API
ClinVar data is fetched via bulk download, not API, because:
- NCBI E-utilities have rate limits (~3 req/sec)
- ClinVar updates monthly (not real-time needs)
- Cluster nodes may have restricted network access

### 2. LatentModules not Callbacks
Callbacks are for side effects (logging, checkpoints). Core pipeline logic belongs in:
- **DataModules**: Data loading and batching
- **LatentModules**: Embedding generation and transformation

### 3. Precomputed Embeddings as DataModule
Multi-channel embeddings (DNA/RNA/Protein) are loaded via `PrecomputedEmbeddingsDataModule` which:
- Loads HDF5 with arbitrary channels
- Provides `get_embeddings() -> Dict[str, Tensor]`
- Handles alignment by variant ID
- Supports label stratification

### 4. Fusion as LatentModule
`PrecomputedFusionModule` is a LatentModule that:
- Takes embeddings from datamodule
- Applies fusion strategy (concat, weighted, attention)
- Returns fused embeddings for downstream metrics

See `design.md` for full rationale.
