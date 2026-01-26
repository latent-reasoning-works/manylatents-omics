# Tasks: Add ClinVar Geometric Pipeline

## 1. Data Infrastructure

- [x] 1.1 Create `scripts/download_clinvar.py` for bulk data preprocessing
- [x] 1.2 Create `manylatents/dogma/data/clinvar_dataset.py` with `ClinVarDataModule`
- [x] 1.3 Create `manylatents/dogma/configs/data/clinvar.yaml` for sequence loading
- [x] 1.4 Create `manylatents/dogma/configs/data/clinvar_precomputed.yaml` for HDF5 loading

## 2. Batch Encoder Algorithm

- [x] 2.1 Create `manylatents/dogma/algorithms/batch_encoder.py`
- [x] 2.2 Create `manylatents/dogma/configs/experiment/clinvar/encode_dna.yaml` (Evo2)
- [x] 2.3 Create `manylatents/dogma/configs/experiment/clinvar/encode_protein.yaml` (ESM3)
- [x] 2.4 Test config dry-run: `python -m manylatents.main --config-name=config --cfg job experiment=clinvar/encode_dna`

## 3. Multi-Channel Embeddings (manylatents core)

**NOTE**: These extend manylatents core capabilities, not dogma-specific.
**PR**: https://github.com/latent-reasoning-works/manylatents/pull/193

- [x] 3.1 Extend `PrecomputedDataModule` for multi-channel support
  - Added `channels: List[str]` parameter
  - Added `get_embeddings() -> Dict[str, Tensor]` method
  - Backward compatible with single-channel usage
- [x] 3.2 Create `MergingModule` (LatentModule)
  - Strategies: concat, weighted_sum, mean
  - Works with multi-channel PrecomputedDataModule or in-memory embeddings
- [x] 3.3 Create `manylatents/dogma/configs/algorithms/latent/merging.yaml`
- [x] 3.4 Create `manylatents/dogma/configs/data/precomputed_embeddings.yaml`

## 4. Geometric Analysis Experiment

- [x] 4.1 Create `manylatents/dogma/configs/experiment/clinvar/geometric_analysis.yaml`
  - Uses MergingModule
  - Configures PR, LID, TSA metrics
- [x] 4.2 Test end-to-end with actual embeddings (GPU encoding)
  - Job 8536281: Evo2 + MergingModule + WandB logging
  - WandB: https://wandb.ai/cesar-valdez-mcgill-university/clinvar-geometric/runs/bqsio66v
  - Note: ESM3 requires HuggingFace gated repo auth

## 5. Orthrus Native Encoder (mamba-ssm 2.x)

**Goal**: Eliminate separate environment requirement for Orthrus.

- [x] 5.1 Re-implement Orthrus MixerModel using mamba-ssm 2.x Block API
  - Import from `mamba_ssm.modules.block` (not `mamba_simple`)
  - Add `mlp_cls=nn.Identity` to Block constructor
- [x] 5.2 Create `manylatents/dogma/encoders/orthrus_native.py`
- [x] 5.3 Update `encoders/__init__.py` to export as `OrthrusEncoder`
- [x] 5.4 Test 3-way fusion: Evo2 (1920) + Orthrus (256) + ESM3 (1536) = 3712-dim
  - Job 8536572: Full E2E test with all 3 live encoders
  - WandB: https://wandb.ai/cesar-valdez-mcgill-university/merging-dogma/runs/imnb1zu7

## 6. Cleanup

- [x] 6.1 Delete `scripts/embed_evo2.py`
- [x] 6.2 Delete `scripts/embed_esm3.py`
- [x] 6.3 Delete `scripts/embed_evo2.sh`
- [x] 6.4 Delete `scripts/embed_esm3.sh`
- [x] 6.5 Delete `scripts/consolidate_embeddings.py`
- [x] 6.6 Delete `scripts/compute_geometric_metrics.py`
- [x] 6.7 Delete `tools/orthrus-embed/` (standalone tool no longer needed)
- [x] 6.8 Delete `scripts/embed_orthrus.sh`
- [x] 6.9 Delete `scripts/precompute_test_orthrus.sh`
- [x] 6.10 Delete `manylatents/dogma/encoders/orthrus.py` (cache-based encoder)

## 7. Testing

- [x] 7.1 Update `tests/dogma/test_encoders.py` for new OrthrusEncoder API
- [x] 7.2 Update `tests/dogma/test_imports.py` for MODEL_CONFIGS
- [x] 7.3 Create `tests/dogma/test_config_e2e.py` (config-only validation)
- [x] 7.4 Fix all experiment configs to use `project: merging-dogma`

## 8. Documentation

- [x] 8.1 Update `CLAUDE.md` with ClinVar pipeline section

## 9. Workshop Experiments (Supervised Analysis)

**Depends on:** manylatents PR #194 (supervised LatentModule support)

- [x] 9.1 Create `configs/experiment/clinvar/baselines.yaml` - single modality classifiers
  - LogisticRegression via ClassifierModule
  - AUC metric for pathogenicity prediction
- [x] 9.2 Create `configs/experiment/clinvar/shared_subspace.yaml` - multi-modal fusion
  - MergingModule with concat_pca strategy
  - LoadingsAnalysisCallback for modality contributions
- [x] 9.3 Create `configs/experiment/clinvar/deviation.yaml` - outlier analysis
  - OutlierScore metric (LOF-based)
  - Tests pathogenicity enrichment in outliers
- [x] 9.4 Create `scripts/run_workshop_experiments.sh` - experiment runner
  - Local and cluster (--cluster) execution modes
  - Runs all baselines, shared subspace, and deviation analyses
