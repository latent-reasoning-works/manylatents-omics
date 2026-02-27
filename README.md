<div align="center">
<pre>
  A T G . . C A T       .  . .
  . G C A T . G .  -->  . .. .  -->  λ(·)
   T . A G C . A T       .  . .
     A T G . C A

     m a n y l a t e n t s - o m i c s

          from sequence to manifold
</pre>

[![license](https://img.shields.io/badge/license-MIT-FDA4AF.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11+-FDA4AF.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/pkg-uv-FDA4AF.svg)](https://docs.astral.sh/uv/)
[![PyPI](https://img.shields.io/badge/PyPI-manylatents--omics-FDA4AF.svg)](https://pypi.org/project/manylatents-omics/)
[![docs](https://img.shields.io/badge/docs-GitHub%20Pages-FDA4AF.svg)](https://latent-reasoning-works.github.io/manylatents-omics/)

</div>

---

Population genetics, single-cell, and foundation model encoders for [manylatents](https://github.com/latent-reasoning-works/manylatents). Extends the core DR framework with biological data types and domain-specific metrics.

## Install

```bash
uv add manylatents-omics
```

Optional extras:

```bash
uv add "manylatents-omics[popgen]"      # population genetics
uv add "manylatents-omics[singlecell]"  # single-cell (scanpy, anndata)
uv add "manylatents-omics[esm]"         # ESM3 protein encoder
uv add "manylatents-omics[orthrus]"     # Orthrus RNA encoder
uv add "manylatents-omics[evo2]"        # Evo2 DNA encoder
```

Or from the core manylatents repo:

```bash
uv sync --extra omics   # installs manylatents-omics as a namespace extension
```

<details>
<summary>development install</summary>

```bash
git clone https://github.com/latent-reasoning-works/manylatents-omics.git
cd manylatents-omics && uv sync
```

</details>

## Architecture

manylatents-omics is a **namespace extension** of [manylatents](https://github.com/latent-reasoning-works/manylatents). It lives alongside the core repo and adds domain-specific modules under the `manylatents.*` namespace via `pkgutil.extend_path()`.

```
lrw/
├── manylatents/    # core DR engine
├── omics/          # this repo — popgen, singlecell, dogma encoders
└── shop/           # cluster infrastructure
```

**Design decision:** The core engine stays domain-agnostic. Each "flavor pack" (omics, vision, etc.) is a separate repo/package that extends the `manylatents` namespace without polluting the core with domain-specific dependencies. Experiment configs (ClinVar pipelines, fusion sweeps, cluster resource presets) belong in downstream experiment repos, not here — this package ships only instantiation configs that define what encoders, datasets, and algorithms *are*.

## Quick start

Omics configs are auto-discovered when the package is installed:

```bash
python -m manylatents.main --config-name=config \
  experiment=single_algorithm data=pbmc_3k

# Sweep on cluster
python -m manylatents.main -m \
  cluster=tamia resources=gpu \
  data=hgdp,pbmc_10k algorithms/latent=umap,phate
```

## Modules

**[popgen](manylatents/popgen/)** — Population genetics via manifold-genetics CSV pipeline. HGDP+1KGP, UK Biobank, All of Us. Admixture proportions, geographic metadata, QC/relatedness filtering. Configs: [`popgen/configs/`](manylatents/popgen/configs/)

**[singlecell](manylatents/singlecell/)** — AnnData `.h5ad` loader for scRNA-seq, scATAC-seq, CITE-seq. Ships with PBMC 3k/10k/68k and Embryoid Body. Any `.h5ad` works via `AnnDataset`. Configs: [`singlecell/configs/`](manylatents/singlecell/configs/)

**[dogma](manylatents/dogma/)** — Foundation model encoders for DNA, RNA, and protein sequences. Supports single-modality encoding, multi-layer extraction, and cross-modal fusion. Configs: [`dogma/configs/`](manylatents/dogma/configs/)

## Encoders

All encoders inherit from [`FoundationEncoder`](manylatents/dogma/encoders/base.py) — lazy model loading, batched encoding with OOM retry, standard `fit()`/`transform()` interface.

- **[ESM3](manylatents/dogma/encoders/esm3.py)** — Protein, 1536-dim, masked mean-pool, true batched forward
- **[Evo2](manylatents/dogma/encoders/evo2.py)** — DNA, 1920/4096/8192-dim (1B/7B/40B), multi-layer extraction, 1M bp context
- **[Orthrus](manylatents/dogma/encoders/orthrus_native.py)** — RNA, 256/512-dim (4-track/6-track), Mamba SSM re-implementation for mamba-ssm 2.x
- **[AlphaGenome](manylatents/dogma/encoders/alphagenome.py)** — DNA, 1536/3072-dim (1bp/128bp), JAX-based, regulatory track predictions, chunked encoding

## Metrics

- **[GeographicPreservation](manylatents/popgen/metrics/preservation.py)** — Spearman correlation between haversine and embedding distances
- **[AdmixturePreservation](manylatents/popgen/metrics/preservation.py)** — Geodesic distance fidelity in admixture simplex vs. latent space
- **[AdmixtureLaplacian](manylatents/popgen/metrics/preservation.py)** — Graph Laplacian smoothness of admixture components over embedding KNN

## ClinVar pipeline

Three-stage variant encoding and geometric analysis. See [docs/clinvar_pipeline.md](docs/clinvar_pipeline.md) for full details. Experiment configs live in downstream repos (e.g. merging_dogma), not in this package.

## Development

```bash
uv sync
pytest tests/ -v
```

## Citing

If manylatents-omics was useful in your research, a citation goes a long way:

```bibtex
@software{manylatents_omics2026,
  title     = {manyLatents-Omics: Biological Extensions for Unified Dimensionality Reduction},
  author    = {Scicluna, Matthew and Valdez C{\'o}rdova, C{\'e}sar Miguel},
  year      = {2026},
  url       = {https://github.com/latent-reasoning-works/manylatents-omics},
  license   = {MIT}
}
```

---

<div align="center">

MIT License · Latent Reasoning Works

</div>
