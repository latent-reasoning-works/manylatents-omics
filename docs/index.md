# manylatents-omics

Biological extensions for [manylatents](https://latent-reasoning-works.github.io/manylatents/) adding population genetics, single-cell omics, and foundation model encoders for DNA, RNA, and protein sequences.

---

## Installation

Install the base package:

```bash
uv add manylatents-omics
```

Enable domain-specific extras depending on your use case:

```bash
# Population genetics (manifold-genetics CSV pipeline)
uv add "manylatents-omics[popgen]"

# Single-cell omics (AnnData / scanpy)
uv add "manylatents-omics[singlecell]"

# Foundation model encoders (ESM3, Orthrus, Evo2 -- requires GPU)
uv add "manylatents-omics[dogma]"
```

For foundation model encoders on CUDA, use [wheelnext uv](https://astral.sh/blog/wheel-variants) to get prebuilt GPU wheels:

```bash
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
uv sync --extra dogma --index-strategy unsafe-best-match
```

---

## Quick Start

Omics configs are auto-discovered when the package is installed:

```bash
# Single-cell: UMAP on PBMC 3k
python -m manylatents.main data=pbmc_3k algorithms/latent=umap

# Population genetics: HGDP dataset
python -m manylatents.main data=hgdp algorithms/latent=phate

# Foundation model encoding: ClinVar DNA
python -m manylatents.main experiment=clinvar/encode_dna
```

!!! note
    Omics configs are auto-discovered when `manylatents-omics` is installed. Just use `python -m manylatents.main` — omics data configs (`data=pbmc_3k`, `data=hgdp`, etc.) will be available automatically.

---

## Modules

manylatents-omics is organized into three domain modules:

| Module | Domain | Data Format | Extra |
|--------|--------|-------------|-------|
| [PopGen](popgen.md) | Population genetics | manifold-genetics CSVs | `[popgen]` |
| [Single-Cell](singlecell.md) | Single-cell omics | AnnData `.h5ad` | `[singlecell]` |
| [Dogma](encoders.md) | DNA / RNA / Protein | FASTA sequences | `[dogma]` |

**PopGen** provides the `ManifoldGeneticsDataModule` for loading PCA, admixture, and geographic data from the manifold-genetics pipeline, along with domain-specific metrics like geographic and admixture preservation.

**Single-Cell** provides `AnnDataModule` for loading scRNA-seq, scATAC-seq, and CITE-seq datasets stored in the AnnData `.h5ad` format. Ships with PBMC 3k, 10k, 68k, and Embryoid Body configs.

**Dogma** provides pretrained foundation model encoders (ESM3, Evo2, Orthrus, AlphaGenome) that transform biological sequences into dense embeddings, plus the [ClinVar pipeline](clinvar_pipeline.md) for multi-modal geometric analysis.

---

## Parent Project

manylatents-omics extends the core [manylatents](https://latent-reasoning-works.github.io/manylatents/) library for dimensionality reduction and geometric analysis. Refer to the parent documentation for details on algorithms, metrics, and the experiment framework.
