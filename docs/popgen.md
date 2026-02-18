# Population Genetics

The `manylatents.popgen` module provides data loading and evaluation metrics for population genetics, built around the [manifold-genetics](https://github.com/latent-reasoning-works/manifold-genetics) CSV pipeline.

---

## Overview

Population genetics studies how genetic variation is distributed across human populations. manylatents-omics loads precomputed PCA coordinates, admixture proportions, and geographic metadata produced by manifold-genetics, then evaluates dimensionality reduction algorithms on how well they preserve population structure.

**Install:** `uv add "manylatents-omics[popgen]"`

---

## Supported Datasets

| Dataset | Populations | Samples | Config |
|---------|-------------|---------|--------|
| HGDP + 1KGP | ~80 | ~4,000 | `data=hgdp` |
| UK Biobank | varies | varies | `data=ukbb` |
| All of Us | varies | varies | `data=aou` |

Datasets are preprocessed upstream by manifold-genetics into a standardized directory structure:

```
output_dir/
  pca/
    fit_pca_*.csv
    transform_pca_*.csv
  admixture/
    fit.K5.csv
    transform.K5.csv
  labels.csv
  colormap.json
```

All CSVs are aligned by a `sample_id` column.

---

## Key Classes

### ManifoldGeneticsDataset

`manylatents.popgen.data.ManifoldGeneticsDataset`

A PyTorch `Dataset` that loads manifold-genetics outputs. It is completely biobank-agnostic -- all preprocessing, filtering, and subsetting is handled upstream by manifold-genetics.

Handles:

- Loading PCA coordinates, admixture ratios, and optional custom embeddings from CSV
- Joining data sources by `sample_id`
- Integer label encoding for PyTorch compatibility (with `get_label_names()` for original strings)
- Geographic coordinate access for preservation metrics

### ManifoldGeneticsDataModule

`manylatents.popgen.data.ManifoldGeneticsDataModule`

A PyTorch Lightning `DataModule` wrapping `ManifoldGeneticsDataset`. Supports two modes:

- **split**: Separate train and test CSVs (from manifold-genetics fit/transform outputs)
- **full**: Same data for both train and test

---

## Metrics

Population genetics metrics evaluate how well a low-dimensional embedding preserves genetic population structure.

| Metric | What it Measures |
|--------|-----------------|
| `GeographicPreservation` | Spearman correlation between haversine distances (lat/lon) and embedding distances |
| `AdmixturePreservation` | Spearman correlation between geodesic admixture distances and embedding distances |
| `AdmixtureLaplacian` | Smoothness of admixture proportions over the embedding's KNN graph (x^T L x) |

All metrics accept an `embeddings` array, the dataset, and optional parameters. They automatically handle subsetting to samples with meaningful geography (excluding recent migrants and admixed populations).

---

## Usage

```bash
# Run UMAP on HGDP population genetics data
python -m manylatents.main data=hgdp algorithms/latent=umap

# Sweep over multiple algorithms
python -m manylatents.main -m \
  data=hgdp \
  algorithms/latent=umap,phate,tsne
```
