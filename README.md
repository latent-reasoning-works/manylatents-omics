# manylatents-omics

Genetics and genomics extension for [manylatents](https://github.com/latent-reasoning-works/manylatents).

## Overview

`manylatents-omics` provides specialized datasets, metrics, and visualizations for analyzing population genetics and genomics data using dimensionality reduction techniques. This package extends the core `manylatents` library with genetics-specific functionality while maintaining a unified API through Python namespace packages.

## Installation

First add the core manylatents package:
```bash
uv add git+https://github.com/latent-reasoning-works/manylatents.git
```

Then add manylatents-omics:
```bash
uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
```

Or add both together:
```bash
uv add git+https://github.com/latent-reasoning-works/manylatents.git git+https://github.com/latent-reasoning-works/manylatents-omics.git
```

Finally, install all packages locally using `sync`.
```bash
uv sync
```

### Foundation Model Encoders (Optional)

Foundation model encoders (ESM3, Orthrus, Evo2) require additional CUDA dependencies that **must be built on an Ampere+ GPU** (compute capability ≥8.0).

| GPU | Compute Capability | Supported |
|-----|-------------------|-----------|
| RTX 8000, RTX 3090 | 7.5 (Turing) | ❌ No |
| A100, A6000 | 8.0 (Ampere) | ✅ Yes |
| L40S, L4 | 8.9 (Ada Lovelace) | ✅ Yes |
| H100, H200 | 9.0 (Hopper) | ✅ Yes |

**Installation on HPC clusters:**
```bash
# 1. Request an Ampere+ GPU allocation
salloc --gpus=l40s:1  # or a100, h100

# 2. Load CUDA module
module load cuda/12.6.0

# 3. Install with optional extras
uv sync --extra esm      # ESM3 only (no special GPU needed)
uv sync --extra orthrus  # Orthrus (needs Ampere+ for mamba-ssm)
uv sync --extra evo2     # Evo2 (needs Ampere+ for transformer-engine)
uv sync --extra dogma    # All foundation encoders
```

**Why?** Packages like `mamba-ssm`, `flash-attn`, and `transformer-engine` use CUDA kernels that require Ampere+ architecture. Building on older GPUs will fail with cryptic nvcc errors.

## Features

### Datasets

#### Single-Cell RNA-seq Datasets
- **AnnDataModule**: LightningDataModule for loading .h5ad files (scanpy format)
  - Supports any AnnData-formatted single-cell dataset
  - Flexible metadata loading via `label_key` parameter
  - Compatible with scRNA-seq, scATAC-seq, CITE-seq, and other single-cell modalities

**Available Datasets:**
- **Embryoid Body (EBD)**: Developmental time series dataset
  - Config: `configs/data/embryoid_body.yaml`
  - Label key: `sample_labels`

- **PBMC (Peripheral Blood Mononuclear Cells)**: 10X Genomics datasets
  - PBMC 3k: ~3,000 cells, ideal for testing and rapid prototyping
    - Config: `configs/data/pbmc_3k.yaml`
  - PBMC 10k: ~10,000 cells, medium-scale dataset (requires manual download)
    - Config: `configs/data/pbmc_10k.yaml`
  - PBMC 68k (reduced): ~700 cells, subsampled version for quick testing/parsing
    - Config: `configs/data/pbmc_68k.yaml`
    - Note: This is a pre-processed reduced dataset. For the full 68k-cell dataset, see instructions below.
  - Label key: `bulk_labels` (cell types), `louvain` (clustering), or `null` for unsupervised learning

**Downloading PBMC Data:**
```bash
# Download all PBMC datasets
python scripts/download_pbmc.py --dataset all --output-dir /path/to/data/single_cell

# Or download a specific size
python scripts/download_pbmc.py --dataset 3k --output-dir /path/to/data/single_cell
```

**Using with Hydra configs:**
```bash
# Use PBMC 3k dataset
python -m manylatents.dogma.main data=pbmc_3k

# Use embryoid body dataset
python -m manylatents.dogma.main data=embryoid_body
```

**Full PBMC 68k Dataset:**

The scanpy version is a reduced dataset with only 700 cells. For the full 68,000-cell dataset:

1. Download from 10X Genomics:
   ```bash
   wget https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices_h5.h5
   ```

2. Convert to h5ad format:
   ```python
   import scanpy as sc
   adata = sc.read_10x_h5('fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices_h5.h5')
   adata.write_h5ad('data/single_cell/pbmc_68k_full.h5ad')
   ```

3. Create a config file `configs/data/pbmc_68k_full.yaml`:
   ```yaml
   _target_: manylatents.singlecell.data.anndata.AnnDataModule
   adata_path: ${paths.data_dir}/single_cell/pbmc_68k_full.h5ad
   label_key: null
   batch_size: 64
   test_split: 0.2
   ```

#### PLINK-based Genetics Datasets
- **PlinkDataset**: Base class for loading PLINK format genetics data
- **HGDP+1KGP**: Human Genome Diversity Project + 1000 Genomes Project
  - `HGDPDataset`: PyTorch Dataset for HGDP data
  - `HGDPData`: LightningDataModule for training pipelines
- **UK Biobank (UKBB)**: Large-scale UK population genetics
  - `UKBBDataset`: PyTorch Dataset for UKBB data
  - `UKBBData`: LightningDataModule for training pipelines
- **All of Us (AOU)**: NIH All of Us Research Program
  - `AOUDataset`: PyTorch Dataset for AOU data
  - `AOUData`: LightningDataModule for training pipelines

All datasets support:
- Train/test splits
- QC filtering
- Relatedness filtering
- Admixture proportion loading
- Geographic metadata
- Precomputed embeddings
- Subsampling

### Metrics

- **GeographicPreservation**: Measures how well embeddings preserve geographic distances between populations
- **AdmixturePreservation**: Evaluates preservation of admixture proportions in embedding space
- **SampleIDMetric**: Maintains sample identity tracking through the analysis pipeline

All metrics support:
- Embedding dimension scaling
- K-curve analysis (preservation vs. neighborhood size)
- Per-sample and aggregate statistics

### Visualizations (Coming Soon)

- **PlotAdmixture**: Visualize embeddings colored by admixture components (PR in progress)

### Scripts

Data preprocessing and analysis scripts:
- `clean_up_admixture.py`: Process admixture analysis outputs
- `make_metadata_file.py`: Generate metadata files for genetics datasets
- `prep_aou_files.py`: Prepare All of Us data files
- `process_admixture.py`: Run admixture analysis pipeline
- `download_data.sh`: Download genetics datasets

## Usage

### Basic Usage

```python
from manylatents.algorithms import PHATE, UMAP
from manylatents.popgen.data import HGDPDataset
from manylatents.popgen.metrics import GeographicPreservation

# Load genetics dataset
dataset = HGDPDataset(
    files={'plink': 'path/to/hgdp_data'},
    cache_dir='./cache',
    data_split='full',
    filter_qc=True,
    filter_related=True
)

# Run dimensionality reduction with core manylatents
algorithm = PHATE(n_components=2)
embeddings = algorithm.fit_transform(dataset)

# Evaluate with genetics-specific metrics
geo_metric = GeographicPreservation(
    embeddings=embeddings,
    dataset=dataset,
    scale_embedding_dimensions=True
)
score = geo_metric.compute()
print(f"Geographic preservation: {score}")
```

### With Lightning DataModule

```python
from manylatents.popgen.data import HGDPData
from manylatents.experiment import DimensionalityReductionExperiment

# Create data module
data = HGDPData(
    files={'plink': 'path/to/hgdp_data'},
    cache_dir='./cache',
    batch_size=256
)

# Run experiment
experiment = DimensionalityReductionExperiment(
    algorithm='phate',
    data=data,
    metrics=['geographic_preservation', 'admixture_preservation']
)
experiment.run()
```

### Configuration with Hydra

#### Data Paths
The project uses a centralized paths configuration (`configs/paths/default.yaml`) to manage data locations:

```yaml
# Default paths - can be overridden via environment variables
data_dir: ${oc.env:MANYLATENTS_DATA_DIR,/network/scratch/c/cesar.valdez/manylatents-omics/data}
cache_dir: ${oc.env:MANYLATENTS_CACHE_DIR,/network/scratch/c/cesar.valdez/manylatents-omics/cache}
output_dir: ${oc.env:MANYLATENTS_OUTPUT_DIR,/network/scratch/c/cesar.valdez/manylatents-omics/outputs}
```

**Directory Structure:**
```
${paths.data_dir}/
├── single_cell/          # Single-cell datasets (scRNA, scATAC, etc.)
│   ├── pbmc_3k.h5ad
│   ├── pbmc_10k.h5ad
│   ├── pbmc_68k.h5ad
│   └── EBT_counts.h5ad
├── HGDP+1KGP/           # Population genetics datasets
│   └── genotypes/
└── ...
```

**Overriding paths:**
```bash
# Via environment variables
export MANYLATENTS_DATA_DIR=/custom/data/path
python -m manylatents.dogma.main data=pbmc_3k

# Via command line
python -m manylatents.dogma.main data=pbmc_3k paths.data_dir=/custom/data/path
```

#### Dataset Configurations

```yaml
# configs/data/hgdp.yaml
_target_: manylatents.popgen.data.HGDPData
files:
  plink: ${paths.data_dir}/HGDP+1KGP/genotypes/hgdp_wgs
cache_dir: ${paths.cache_dir}
data_split: full
filter_qc: true
filter_related: true
batch_size: 256

# configs/metrics/geographic_preservation.yaml
_target_: manylatents.popgen.metrics.GeographicPreservation
scale_embedding_dimensions: true
k_values: [5, 10, 20, 50]
```

## Data Format

### PLINK Files
Genetics datasets expect PLINK binary format:
- `.bed`: Genotype data (binary)
- `.bim`: SNP information
- `.fam`: Sample information

### Metadata
CSV files with required columns:
- `sample_id`: Sample identifier
- `Population`: Population label
- `latitude`: Geographic latitude
- `longitude`: Geographic longitude
- Additional QC and filtering columns (dataset-specific)

### Admixture Files
TSV files from ADMIXTURE software:
- `K{N}.Q`: Admixture proportions for K={N} ancestral populations

## Project Structure

```
manylatents-omics/
├── manylatents/
│   ├── __init__.py           # Namespace package
│   └── omics/
│       ├── __init__.py       # Main omics module
│       ├── data/             # Genetics datasets
│       │   ├── plink_dataset.py
│       │   ├── hgdp_dataset.py
│       │   ├── aou_dataset.py
│       │   └── ukbb_dataset.py
│       ├── metrics/          # Genetics metrics
│       │   ├── preservation.py
│       │   └── sample_id.py
│       ├── callbacks/        # Visualizations (coming soon)
│       └── scripts/          # Data processing scripts
├── configs/                  # Hydra configuration files
│   ├── data/
│   └── metrics/
├── tests/                    # Unit tests
└── pyproject.toml           # Package configuration
```

## Architecture

`manylatents-omics` uses Python namespace packages to extend `manylatents` with genetics-specific functionality:

- **Core manylatents** (`manylatents.*`): Algorithms, base datasets, core metrics
- **Omics extension** (`manylatents.popgen.*`): Genetics datasets, preservation metrics

This architecture provides:
- **Separation of concerns**: Core DR algorithms vs. genetics applications
- **Independent maintenance**: Genetics team owns omics repository
- **Unified API**: Import from `manylatents.popgen.*` alongside core `manylatents.*`
- **Optional installation**: Install omics extension only when needed

## Citation

If you use manylatents-omics in your research, please cite:

```bibtex
@software{manylatents_omics,
  title = {manylatents-omics: Genetics Extension for manylatents},
  author = {Latent Reasoning Works},
  year = {2025},
  url = {https://github.com/latent-reasoning-works/manylatents-omics}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- **Issues**: https://github.com/latent-reasoning-works/manylatents-omics/issues
- **Discussions**: https://github.com/latent-reasoning-works/manylatents-omics/discussions
- **Core manylatents**: https://github.com/latent-reasoning-works/manylatents
