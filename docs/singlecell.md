# Single-Cell Omics

The `manylatents.singlecell` module provides data loading for single-cell omics datasets stored in the AnnData `.h5ad` format, covering scRNA-seq, scATAC-seq, and CITE-seq assays.

---

## Overview

Single-cell experiments measure gene expression (or chromatin accessibility, surface proteins, etc.) in individual cells. The resulting count matrices are stored as AnnData objects in `.h5ad` files, which manylatents-omics loads directly into PyTorch for dimensionality reduction and geometric analysis.

**Install:** `uv add "manylatents-omics[singlecell]"`

---

## Shipped Datasets

Preconfigured Hydra configs are provided for common benchmark datasets:

| Dataset | Cells | Features | Config |
|---------|-------|----------|--------|
| PBMC 3k | ~2,700 | ~1,800 genes | `data=pbmc_3k` |
| PBMC 10k | ~10,000 | varies | `data=pbmc_10k` |
| PBMC 68k | ~68,000 | varies | `data=pbmc_68k` |
| Embryoid Body | varies | varies | `data=embryoid_body` |
| LARRY in vitro (Weinreb et al. 2020) | ~130,000 with a clone | 2,000 HVGs (+50 PCs) | `data=larry` |

LARRY is built on first use: `LarryDataModule` (or `manylatents.singlecell.data.larry.load_larry()`) downloads the public in-vitro files from the paper website (about 2.1 GB) into the omics data root and writes `single_cell/larry.h5ad`. The module docstring documents how each column maps to the source. `manylatents.singlecell.sampling.clone_subsample(adata.obs, n)` subsamples whole clones: clones with a day-2 cell and at least three day-4/6 descendants, in seeded random order, up to `n` cells.

---

## Key Classes

### AnnDataset

`manylatents.singlecell.data.AnnDataset`

A PyTorch `Dataset` for any AnnData `.h5ad` file. Supports:

- Loading from `adata.X`, `adata.raw.X`, or a named layer (`adata.layers[layer]`)
- Automatic sparse-to-dense conversion
- Cell-type label extraction from `adata.obs` with integer encoding
- Access to observation annotations via `get_obs(key)`

### AnnDataModule

`manylatents.singlecell.data.AnnDataModule`

A PyTorch Lightning `DataModule` wrapping `AnnDataset`. Supports two modes:

- **full**: Entire dataset used for both training and testing
- **split**: Random train/test split with configurable ratio and seed

---

## Usage

```bash
# UMAP on PBMC 3k
python -m manylatents.main data=pbmc_3k algorithms/latent=umap

# Sweep datasets and algorithms
python -m manylatents.main -m \
  data=pbmc_3k,pbmc_10k \
  algorithms/latent=umap,phate
```

### Loading a Custom Dataset

To use your own `.h5ad` file, create a Hydra config or instantiate the datamodule directly:

```python
from manylatents.singlecell.data import AnnDataModule

dm = AnnDataModule(
    adata_path="path/to/your_data.h5ad",
    label_key="cell_type",
    batch_size=128,
    mode="full",
)
dm.setup()
```
