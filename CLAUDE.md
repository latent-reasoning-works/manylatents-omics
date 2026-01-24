<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Manylatents-Omics - Multi-Omics Datasets and Foundation Model Encoders

This repository extends the manyLatents dimensionality reduction framework with specialized datasets and encoders for omics data, including population genetics, single-cell omics, and central dogma sequences (DNA/RNA/Protein).

---

## Entry Point

**Always use the omics entry point** to ensure Hydra discovers omics configs:

```bash
# Load CUDA on HPC clusters first
module load anaconda/3 cuda/12.4.1

# Run experiments
python -m manylatents.omics.main --config-name=config experiment=single_algorithm
python -m manylatents.omics.main --config-name=config experiment=central_dogma_fusion
```

**Why `manylatents.omics.main` instead of `manylatents.main`?**

The omics entry point registers `OmicsSearchPathPlugin` before Hydra initializes, making dogma/popgen/singlecell configs available on the search path. The standard `manylatents.main` doesn't know about omics configs.

See `docs/README.md` for detailed installation and usage guide.

---

## Repository Architecture

### Three-Module Structure

| Module | Domain | Data Types | Key Datasets |
|--------|--------|------------|--------------|
| **`manylatents.popgen`** | Population genetics | PLINK-based genotype data | HGDP, AOU, UKBB, MHI |
| **`manylatents.singlecell`** | Single-cell omics | AnnData (.h5ad) | PBMC (3k/10k/68k-reduced), Embryoid Body |
| **`manylatents.dogma`** | Central dogma sequences | FASTA/DNA/RNA/Protein | GFP, synthetic sequences |

**Design Principle**: Separation by biological domain and data format.

---

## Module Details

### `manylatents.popgen` - Population Genetics

**Focus**: Human population genomics using genotype data from large biobanks.

**Data Format**: PLINK (.bed/.bim/.fam)

**Available Datasets**:
- **HGDP** (Human Genome Diversity Project) - Global population structure
- **AOU** (All of Us) - US diverse cohort
- **UKBB** (UK Biobank) - Large-scale European cohort
- **MHI** (Million Health Initiative) - Additional biobank data

**Key Components**:
- `PlinkDataset` - Core PLINK data loader
- Dataset-specific modules: `HGDPDataModule`, `AOUDataModule`, `UKBBDataModule`, `MHIDataModule`
- Population genetics metrics: `GeographicPreservation`, `AdmixturePreservation`
- Callbacks: `PlotAdmixture` for visualization

**Example Usage**:
```bash
# HGDP experiment
python -m manylatents.omics.main --config-name=config experiment=single_algorithm data=hgdp algorithms/latent=umap
```

---

### `manylatents.singlecell` - Single-Cell Omics

**Focus**: Single-cell RNA-seq, ATAC-seq, and multimodal single-cell data.

**Data Format**: AnnData (.h5ad)

**Available Datasets**:
- **PBMC 3k** - ~3,000 cells, testing/rapid prototyping
- **PBMC 10k** - ~10,000 cells, medium-scale experiments (requires manual download)
- **PBMC 68k (reduced)** - ~700 cells, subsampled version for quick testing/parsing
  - Note: This is a pre-processed reduced dataset from scanpy. For the full 68k-cell dataset, see README.md.
- **Embryoid Body** - Developmental time series with cell type labels

**Key Components**:
- `AnnDataset` - PyTorch Dataset for AnnData objects
- `AnnDataModule` - Lightning DataModule with train/val/test splits
- Supports cell type labels via `label_key` parameter
- Handles raw counts, normalized data, and custom layers

**Example Usage**:
```bash
# PBMC unsupervised learning
python -m manylatents.omics.main --config-name=config experiment=single_algorithm data=pbmc_10k algorithms/latent=phate

# Embryoid body with labels
python -m manylatents.omics.main --config-name=config experiment=single_algorithm data=embryoid_body algorithms/latent=umap
```

**Config Structure**:
```yaml
# configs/data/pbmc_10k.yaml
_target_: manylatents.singlecell.data.anndata.AnnDataModule
adata_path: ${paths.data_dir}/single_cell/pbmc_10k.h5ad
label_key: null  # or 'cell_type', 'sample_labels', etc.
layer: null      # or 'raw_counts', 'normalized', etc.
use_raw: false
batch_size: 128
test_split: 0.2
```

---

### `manylatents.dogma` - Central Dogma Sequences

**Focus**: Foundation model encoders for DNA, RNA, and protein sequences.

**Data Format**: FASTA, raw sequences

**Available Encoders**:

| Encoder | Domain | Model Size | Embedding Dim | Use Case |
|---------|--------|------------|---------------|----------|
| **ESM3** | Protein | 1.4B params | 1536 | Protein structure/function |
| **Evo2** | DNA | StripedHyena2 | 2048 | Long-range genomic context |
| **Orthrus** | RNA | Mamba-based | 256/512 | RNA secondary structure |

**Key Components**:
- Foundation model wrappers in `manylatents.dogma.encoders`
- `SequenceDataModule` for FASTA/sequence data
- GPU-accelerated inference (see GPU requirements below)

**Example Usage**:
```bash
# Encode proteins with ESM3
python -m manylatents.dogma.encode encoder=esm3 data=sequence_gfp

# DNA sequences with Evo2
python -m manylatents.dogma.encode encoder=evo2 data=sequence_synthetic
```

**GPU Requirements**:
- **ESM3**: 16GB+ VRAM (batch_size=8)
- **Evo2**: 24GB+ VRAM (long sequences), Ampere+ GPU (A100/L40S/H100) for FP8
- **Orthrus**: 8GB+ VRAM (efficient Mamba architecture)

**Installation** (requires wheelnext uv for prebuilt CUDA wheels):
```bash
# 1. Install wheelnext uv (one-time, replaces standard uv)
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh

# 2. Lock and sync dogma extras (use unsafe-best-match for cross-index resolution)
uv lock --index-strategy unsafe-best-match
uv sync --extra dogma --index-strategy unsafe-best-match

# 3. Verify imports (requires CUDA module on login nodes)
module load cuda/12.4.1
uv run python -c "import evo2, orthrus, esm; print('All encoders OK')"
```

**Why wheelnext?** Standard uv/pip can't find prebuilt wheels for transformer-engine-torch and mamba-ssm (they're on conda-forge, not PyPI). Wheelnext uv supports [wheel variants](https://astral.sh/blog/wheel-variants) which auto-detect GPU and select CUDA-compatible wheels.

**Why `--index-strategy unsafe-best-match`?** torch 2.7.x is on PyPI, but the wheelnext PyTorch index only has 2.8.0. This flag allows uv to consider all indexes for version resolution.

---

## Architectural Decision: Issue 12 Refactoring

### Why Separate `popgen` and `singlecell`?

**Original Problem** (Issue #12): AnnData code was located in `manylatents.popgen.data`, but anndata is the standard format for **single-cell omics**, not population genetics.

**Solution**: Created `manylatents.singlecell` module to properly separate:
- **Population genetics** (PLINK genotypes) → `popgen`
- **Single-cell omics** (AnnData) → `singlecell`
- **Sequence data** (DNA/RNA/Protein) → `dogma`

**Migration Details**:
- Moved `anndata.py` and `anndata_dataset.py` from `popgen.data` to `singlecell.data`
- Updated all configs: `pbmc_3k.yaml`, `pbmc_10k.yaml`, `pbmc_68k.yaml`, `embryoid_body.yaml`
- Updated imports in `popgen/data/__init__.py` to remove single-cell references

**Benefits**:
- Clear conceptual boundaries
- Easy to extend (e.g., add ATAC-seq to singlecell, new biobanks to popgen)
- Avoids confusion for new contributors
- Aligns with biological domain expertise

**Implementation**: This refactoring followed LRW's Shop dev-workflow (see below).

---

## Development Guidelines

### Adding New Datasets

**For Population Genetics (PLINK data)**:
1. Create dataset class inheriting from `PlinkDataset`
2. Create Lightning DataModule in `manylatents/popgen/data/`
3. Add config to `configs/data/your_dataset.yaml`
4. Update `manylatents/popgen/data/__init__.py` exports

**For Single-Cell (AnnData)**:
1. Prepare `.h5ad` file with standard AnnData structure
2. Create config in `configs/data/your_dataset.yaml`:
   ```yaml
   _target_: manylatents.singlecell.data.anndata.AnnDataModule
   adata_path: ${paths.data_dir}/single_cell/your_data.h5ad
   label_key: cell_type  # or null for unsupervised
   ```
3. No new Python code needed - `AnnDataModule` is generic

**For Sequence Data**:
1. Prepare FASTA file or sequence CSV
2. Create config referencing appropriate encoder:
   ```yaml
   _target_: manylatents.dogma.data.SequenceDataModule
   sequence_file: ${paths.data_dir}/sequences/your_seqs.fasta
   ```

### Config Composition Patterns

Use Hydra's composition to combine data, algorithm, and cluster configs:

```bash
# Local testing
python -m manylatents.omics.main data=pbmc_3k algorithm=umap

# Cluster submission (via Shop)
python -m manylatents.omics.main -m \
  cluster=mila_remote \
  resources=gpu \
  data=pbmc_68k \
  algorithm=phate,umap,diffusion_map
```

### Testing Patterns

**Import Tests**: Ensure all modules can be imported
```python
from manylatents.singlecell.data import AnnDataModule, AnnDataset
from manylatents.popgen.data import HGDPDataModule, PlinkDataset
```

**Config Tests**: Verify Hydra can parse configs
```bash
python -m manylatents.omics.main --cfg job data=pbmc_10k
```

**E2E Tests**: Run with minimal resources
```bash
python -m manylatents.omics.main data=pbmc_3k cluster=local resources=test
```

---

<!-- BEGIN LRW SHARED INFRASTRUCTURE -->
## LRW Shared Infrastructure

### Cluster Job Submission

Use shop's Hydra launcher for all sweep jobs:

```bash
uv run python -m manylatents.omics.main -m \
  cluster=mila_remote \
  resources=gpu \
  data=pbmc_68k \
  algorithm=umap,phate,tsne
```

See `../shop/CLAUDE.md` for complete documentation on:
- Available clusters (mila, mila_remote, narval, cedar)
- Resource profiles (cpu, gpu, cpu_high_mem)
- WandB integration
- Monitoring jobs

### Shop Agents (Development Workflow)

Use the integrated agent pipeline for quality gates:

| Agent | Purpose |
|-------|---------|
| `dev-workflow` | Meta agent - orchestrates full pipeline |
| `plan-auditor` | Validate ecosystem adherence, config composition |
| `adversarial-tester` | Design tests that break assumptions |
| `code-simplifier` | Refactor with coverage gates |

**Workflow**: plan-auditor → adversarial-tester → implement → code-simplifier

**Example**: The Issue 12 refactoring (separating singlecell from popgen) was designed with Shop's dev-workflow principles:
1. **Plan Audit**: Verified no broken imports, config composition still valid
2. **Test Design**: Checked import paths, config instantiation, file moves
3. **Implementation**: Git moves to preserve history, systematic config updates
4. **Verification**: Confirmed all configs reference correct module paths

### Testing Requirements

- **Coverage threshold**: 80% minimum
- **Critical paths**: 100% coverage (cluster jobs, WandB logging, cross-repo interfaces)
- **E2E tests**: Use `cluster=local resources=test`

```bash
# Run tests with coverage
pytest --cov=manylatents --cov-fail-under=80
```

### Config Templates

Copy from `../shop/shop/hydra/config_templates/` to your configs:
- `cluster/` → `configs/cluster/`
- `resources/` → `configs/resources/`
- `tests/` → `tests/`
<!-- END LRW SHARED INFRASTRUCTURE -->

---

## Data Directory Structure

Centralized paths configuration (see `configs/paths/default.yaml`):

```
${MANYLATENTS_DATA_DIR}/  # Default: ./data/
├── single_cell/
│   ├── pbmc_3k.h5ad
│   ├── pbmc_10k.h5ad
│   ├── pbmc_68k.h5ad
│   └── EBT_counts.h5ad
├── popgen/
│   ├── hgdp/
│   ├── aou/
│   ├── ukbb/
│   └── mhi/
└── sequences/
    ├── gfp.fasta
    └── synthetic.csv
```

Set `MANYLATENTS_DATA_DIR` environment variable to override default location.

---

## Quick Start Examples

### Single-Cell Analysis

```bash
# Download PBMC data
python scripts/download_pbmc.py

# Run UMAP on PBMC 10k
python -m manylatents.omics.main data=pbmc_10k algorithm=umap

# Cluster sweep with multiple algorithms
python -m manylatents.omics.main -m \
  cluster=mila_remote \
  resources=gpu \
  data=pbmc_68k \
  algorithm=umap,phate,diffusion_map
```

### Population Genetics

```bash
# HGDP analysis
python -m manylatents.omics.main data=hgdp algorithm=pca

# Sweep across datasets
python -m manylatents.omics.main -m \
  cluster=mila_remote \
  resources=cpu_high_mem \
  data=hgdp,aou,ukbb \
  algorithm=umap
```

### Sequence Encoding

```bash
# Encode proteins with ESM3
python -m manylatents.dogma.encode encoder=esm3 data=sequence_gfp

# Requires GPU with 16GB+ VRAM
python -m manylatents.dogma.encode \
  encoder=esm3 \
  cluster=mila_remote \
  resources=gpu
```

### Central Dogma Fusion (DNA + RNA + Protein)

All three encoders work in the same environment - no separate setup needed.

```bash
# Run E2E fusion test (GPU required)
sbatch scripts/run_e2e_test.sh

# Or via Hydra
python -m manylatents.omics.main --config-name=config experiment=central_dogma_fusion
```

**Encoders** (all mamba-ssm 2.x compatible):
| Encoder | Modality | Embedding Dim | HuggingFace |
|---------|----------|---------------|-------------|
| **Evo2Encoder** | DNA | 1920 | arcinstitute/evo2_1b_base |
| **OrthrusEncoder** | RNA | 256 | quietflamingo/orthrus-base-4-track |
| **ESM3Encoder** | Protein | 1536 | esm3_sm_open_v1 |

**Output**: 3712-dim fused embeddings (via MergingModule concat)

**Implementation Note**: OrthrusEncoder was re-implemented to use mamba-ssm 2.x Block API
(`mamba_ssm.modules.block.Block` with `mlp_cls=nn.Identity`), eliminating the version
conflict with Evo2 that previously required a separate environment.

---

### ClinVar Variant Analysis Pipeline

Hydra-composed pipeline for encoding ClinVar pathogenic/benign variants and computing geometric metrics on fused embeddings.

#### Prerequisites

1. **Download ClinVar data** (one-time):
```bash
python scripts/download_clinvar.py --genes BRCA1,BRCA2 --output data/clinvar/
```

This downloads variant_summary from NCBI FTP and fetches sequences from Ensembl REST API.

#### Step 1: Encode DNA with Evo2

```bash
# Local (requires L40S/H100)
python -m manylatents.omics.main --config-name=config experiment=clinvar/encode_dna

# Limit variants for testing
python -m manylatents.omics.main --config-name=config experiment=clinvar/encode_dna data.max_variants=100

# Cluster submission
python -m manylatents.omics.main --config-name=config experiment=clinvar/encode_dna \
  cluster=mila_remote resources=gpu
```

Output: `${paths.output_dir}/embeddings/clinvar/evo2.pt`

#### Step 2: Encode Protein with ESM3

```bash
python -m manylatents.omics.main --config-name=config experiment=clinvar/encode_protein

# Cluster submission (can run in parallel with DNA)
python -m manylatents.omics.main --config-name=config experiment=clinvar/encode_protein \
  cluster=mila_remote resources=gpu
```

Output: `${paths.output_dir}/embeddings/clinvar/esm3.pt`

#### Step 3: Geometric Analysis on Fused Embeddings

```bash
# Fuse embeddings and compute PR, LID, TSA metrics
python -m manylatents.omics.main --config-name=config experiment=clinvar/geometric_analysis

# With weighted fusion
python -m manylatents.omics.main --config-name=config experiment=clinvar/geometric_analysis \
  algorithms.latent.strategy=weighted_sum \
  'algorithms.latent.weights={evo2: 0.5, esm3: 0.5}'
```

#### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ClinVarDataModule` | `manylatents.dogma.data` | Load ClinVar sequences by modality |
| `BatchEncoder` | `manylatents.dogma.algorithms` | Encode sequence batches with foundation models |
| `MergingModule` | `manylatents.algorithms.latent` | Fuse multi-channel embeddings (concat, weighted_sum, mean) |
| `PrecomputedDataModule` | `manylatents.data` | Load precomputed embeddings with `channels` param |

#### Experiment Configs

- `experiment=clinvar/encode_dna` - Evo2 DNA encoding
- `experiment=clinvar/encode_protein` - ESM3 protein encoding
- `experiment=clinvar/geometric_analysis` - Fuse embeddings + compute metrics

#### Multi-Channel Embedding Support

The pipeline uses manylatents core's multi-channel embedding support:

```python
# Load multiple embedding channels
dm = PrecomputedDataModule(
    path="embeddings/clinvar/",
    channels=["evo2", "esm3"],
)
dm.setup()
embs = dm.get_embeddings()  # {"evo2": Tensor, "esm3": Tensor}

# Fuse embeddings
merger = MergingModule(strategy="concat", datamodule=dm)
fused = merger.fit_transform(dummy)
```

See [PR #193](https://github.com/latent-reasoning-works/manylatents/pull/193) for implementation details.

---

## Issue Tracking

- **Issue 12** ✅ RESOLVED: Refactored anndata out of popgen into new singlecell module
- **Issue 13** 🔄 OPEN: Refactor popgen datasets to consume manifold-genetics outputs (out of scope - external codebase)

---

## Contributing

This repository follows LRW development practices:
1. Use Shop's dev-workflow agent for significant changes
2. Maintain 80% test coverage (100% for critical paths)
3. Test configs with `--cfg job` before cluster submission
4. Document architectural decisions in this file

---

## Additional Resources

- **Shop**: `../shop/` - LRW shared infrastructure
- **Core manyLatents**: Main dimensionality reduction framework
- **WandB Logging**: Integrated via Shop's launchers
- **GPU Cluster Access**: See `../shop/CLAUDE.md` for cluster configs
