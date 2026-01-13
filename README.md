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
## Features

### Datasets

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
