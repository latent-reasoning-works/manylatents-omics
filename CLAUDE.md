# Manylatents-Omics - Multi-Omics Datasets and Foundation Model Encoders

This repository extends the manyLatents dimensionality reduction framework with specialized datasets and encoders for omics data, including population genetics, single-cell omics, and central dogma sequences (DNA/RNA/Protein).

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

**Focus**: Human population genomics using precomputed outputs from manifold-genetics.

**Data Format**: CSV files from manifold-genetics pipeline (PCA, admixture, labels)

**New Interface (Recommended)**:
- **ManifoldGeneticsDataset** - Dataset-agnostic loader for manifold-genetics outputs
- **ManifoldGeneticsDataModule** - Lightning DataModule for manifold-genetics data
- Works with any biobank: HGDP, AOU, UKBB, MHI, or custom cohorts
- No preprocessing logic - all filtering done upstream by manifold-genetics

**Legacy Interface (Deprecated)**:
- **PlinkDataset** - Base class for raw PLINK data loading
- Dataset-specific modules: `HGDPDataModule`, `AOUDataModule`, `UKBBDataModule`, `MHIDataModule`
- Contains biobank-specific filtering and preprocessing logic
- Will be removed in future version

**Key Components**:
- Population genetics metrics: `GeographicPreservation`, `AdmixturePreservation`
- Callbacks: `PlotAdmixture`, `PlotEmbeddings` for visualization

**Example Usage (New Interface)**:
```bash
# Using manifold-genetics outputs
python -m manylatents.main data=manifold_genetics_hgdp algorithm=umap
```

**Example Usage (Legacy Interface)**:
```bash
# Using raw PLINK data (deprecated)
python -m manylatents.main data=hgdp algorithm=umap
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
python -m manylatents.main data=pbmc_10k algorithm=phate

# Embryoid body with labels
python -m manylatents.main data=embryoid_body algorithm=umap
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
- **Evo2**: 24GB+ VRAM (long sequences)
- **Orthrus**: 8GB+ VRAM (efficient Mamba architecture)

---

## Architectural Decision: Issue 13 Refactoring

### Migrating from PLINK-based Datasets to manifold-genetics Interface

**Problem**: The original dataset layer was tightly coupled to raw PLINK ingestion with dataset-specific preprocessing (UKBB, AoU, HGDP, MHI). This resulted in:
- Significant code duplication across biobank-specific classes
- Complex filtering logic (filter_qc, filter_related, remove_recent_migration, etc.)
- Tight coupling between data loading and domain-specific ETL
- Adding new cohorts required duplicating code

**Solution**: Created `ManifoldGeneticsDataset` and `ManifoldGeneticsDataModule` that consume standardized outputs from the [manifold-genetics](https://github.com/MattScicluna/manifold_genetics) pipeline.

### manifold-genetics Output Format

Expected directory structure:
```
output_dir/
├── pca/
│   ├── fit_pca_50.csv       # PCA coordinates for training samples
│   └── transform_pca_50.csv # PCA coordinates for test samples
├── admixture/
│   ├── fit.K5.csv           # Admixture proportions (K=5) for training
│   ├── transform.K5.csv     # Admixture proportions (K=5) for test
│   ├── fit.K7.csv           # Multiple K values supported
│   └── transform.K7.csv
├── embeddings/              # Optional custom embeddings
│   ├── fit_embedding.csv
│   └── transform_embedding.csv
├── labels.csv               # Sample metadata with sample_id column
└── colormap.json            # Label-to-color mapping for visualization
```

All CSVs must have a `sample_id` column for alignment.

**CSV Format Requirements**:
- **PCA CSV**: `sample_id, dim_1, dim_2, ..., dim_n`
  - Example: `HG00096,0.07330787,0.212584,-0.01297431,...`
- **Admixture CSV**: `sample_id, component_1, component_2, ..., component_K`
  - Components sum to 1.0 per sample
  - Example: `HGDP00001,0.9996,0.0004`
- **Labels CSV**: `sample_id, <label_column>, [other label columns], [latitude, longitude]`
  - Example:
    ```
    sample_id,Population,Genetic_region,latitude,longitude
    HGDP00001,Yoruba,Africa,6.5244,3.3792
    HGDP00002,Yoruba,Africa,6.5244,3.3792
    HGDP00003,Han,EastAsia,39.9042,116.4074
    ```
- **Embeddings CSV**: `sample_id, dim_1, dim_2, ..., dim_n` (or custom column names)
- **Colormap JSON**: Nested dict by label type:
  ```json
  {
    "Population": {
      "Yoruba": "#FF0000",
      "Han": "#00FF00"
    },
    "Genetic_region": {
      "Africa": "#FF6B6B",
      "EastAsia": "#4ECDC4"
    }
  }
  ```
  Each label type (Population, Genetic_region) maps to a dict of label values to hex colors.

### Migration Example

**Before (Legacy PLINK-based)**:
```yaml
# configs/data/hgdp_old.yaml
_target_: manylatents.popgen.data.HGDPDataModule
files:
  plink: ./data/HGDP/genotypes/...
  metadata: ./data/HGDP/metadata.csv
  admixture: ./data/HGDP/admixture/global.{K}_metadata.tsv
  admixture_K: 2,3,4,5
filter_qc: True
filter_related: False
test_all: True
remove_recent_migration: False
```

**After (manifold-genetics based)**:
```yaml
# configs/data/hgdp_new.yaml
_target_: manylatents.popgen.data.ManifoldGeneticsDataModule
fit_pca_path: ./data/HGDP/manifold_genetics/pca/fit_pca_50.csv
transform_pca_path: ./data/HGDP/manifold_genetics/pca/transform_pca_50.csv
fit_admixture_paths:
  5: ./data/HGDP/manifold_genetics/admixture/fit.K5.csv
transform_admixture_paths:
  5: ./data/HGDP/manifold_genetics/admixture/transform.K5.csv
labels_path: ./data/HGDP/manifold_genetics/labels.csv
colormap_path: ./data/HGDP/manifold_genetics/colormap.json
label_column: Population
```

**Key Changes**:
- ✅ Removed: `filter_qc`, `filter_related`, `remove_recent_migration` - handled upstream
- ✅ Removed: Dataset-specific class inheritance (HGDPDataModule → ManifoldGeneticsDataModule)
- ✅ Added: Explicit paths to fit/transform CSVs
- ✅ Added: `colormap.json` for consistent visualization
- ✅ Simplified: All filtering done once by manifold-genetics, not repeated in manylatents

### Benefits

- **Separation of concerns**: manifold-genetics handles ETL/genomics, manylatents handles representation learning
- **No code duplication**: Single dataset class works for all biobanks
- **Easy to add new cohorts**: Just run manifold-genetics pipeline and update config paths
- **Inspectable data**: CSV format is human-readable and easy to debug
- **Consistent visualization**: colormap.json ensures consistent colors across experiments
- **Faster development**: Adding HGDP, AOU, UKBB, or custom cohort is now a config problem, not a code problem

### Deprecation Timeline

- **Current**: Both legacy (PLINK-based) and new (manifold-genetics) interfaces available
- **Next Release**: Add deprecation warnings to legacy classes
- **Future**: Remove legacy PlinkDataset, HGDPDataset, AOUDataset, UKBBDataset, MHIDataset, precomputed_mixin.py

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

**For Population Genetics (Recommended - manifold-genetics)**:
1. Run the manifold-genetics pipeline on your cohort to generate outputs:
   - PCA coordinates (fit and transform CSVs)
   - Admixture proportions (fit and transform CSVs for each K)
   - labels.csv with sample metadata
   - colormap.json for visualization
2. Create config in `configs/data/your_cohort.yaml`:
   ```yaml
   _target_: manylatents.popgen.data.ManifoldGeneticsDataModule
   fit_pca_path: ${paths.data_dir}/your_cohort/pca/fit_pca_50.csv
   transform_pca_path: ${paths.data_dir}/your_cohort/pca/transform_pca_50.csv
   fit_admixture_paths:
     5: ${paths.data_dir}/your_cohort/admixture/fit.K5.csv
   transform_admixture_paths:
     5: ${paths.data_dir}/your_cohort/admixture/transform.K5.csv
   labels_path: ${paths.data_dir}/your_cohort/labels.csv
   colormap_path: ${paths.data_dir}/your_cohort/colormap.json
   label_column: Population
   batch_size: 128
   num_workers: 4
   ```
3. No Python code needed - config changes only!

**For Population Genetics (Legacy - PLINK data, deprecated)**:
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
python -m manylatents.main data=pbmc_3k algorithm=umap

# Cluster submission (via Shop)
python -m manylatents.main -m \
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
python -m manylatents.main --cfg job data=pbmc_10k
```

**E2E Tests**: Run with minimal resources
```bash
python -m manylatents.main data=pbmc_3k cluster=local resources=test
```

---

<!-- BEGIN LRW SHARED INFRASTRUCTURE -->
## LRW Shared Infrastructure

### Cluster Job Submission

Use shop's Hydra launcher for all sweep jobs:

```bash
uv run python -m manylatents.main -m \
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
python -m manylatents.main data=pbmc_10k algorithm=umap

# Cluster sweep with multiple algorithms
python -m manylatents.main -m \
  cluster=mila_remote \
  resources=gpu \
  data=pbmc_68k \
  algorithm=umap,phate,diffusion_map
```

### Population Genetics

```bash
# HGDP analysis
python -m manylatents.main data=hgdp algorithm=pca

# Sweep across datasets
python -m manylatents.main -m \
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

---

## Issue Tracking

- **Issue 12** ✅ RESOLVED: Refactored anndata out of popgen into new singlecell module
- **Issue 13** ✅ IN PROGRESS: Refactor popgen datasets to consume manifold-genetics outputs
  - ✅ Created ManifoldGeneticsDataset and ManifoldGeneticsDataModule
  - ✅ Updated PlotEmbeddings callback to support colormap.json
  - ✅ Added comprehensive tests and example config
  - ✅ Updated documentation with migration guide
  - ⏳ Legacy classes still available for backward compatibility
  - 🔜 Future: Add deprecation warnings and remove legacy classes

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
