<div align="center">
<pre>
  A T G . . C A T       .  . .
  . G C A T . G .  -->  . .. .  -->  λ(·)
   T . A G C . A T       .  . .
     A T G . C A

  m a n y l a t e n t s - o m i c s

       from sequence to manifold
</pre>

[![license](https://img.shields.io/badge/license-MIT-2DD4BF.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11+-2DD4BF.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/pkg-uv-2DD4BF.svg)](https://docs.astral.sh/uv/)

</div>

---

Population genetics, single-cell, and foundation model encoders for [manylatents](https://github.com/latent-reasoning-works/manylatents). Extends the core DR framework with biological data types and domain-specific metrics.

## Modules

| Module | Domain | Data | Config example |
|--------|--------|------|----------------|
| `manylatents.popgen` | Population genetics | PLINK binary (.bed/.bim/.fam) | `data=hgdp` |
| `manylatents.singlecell` | Single-cell omics | AnnData (.h5ad) | `data=pbmc_3k` |
| `manylatents.dogma` | DNA / RNA / Protein | FASTA, sequences | `encoder=esm3` |

## Install

```bash
# Base (popgen + singlecell)
git clone https://github.com/latent-reasoning-works/manylatents-omics.git
cd manylatents-omics
uv sync

# Foundation model encoders (requires wheelnext uv + Ampere+ GPU)
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
uv sync --extra dogma
```

## Quick start

Always use the omics entry point — it registers omics configs on the Hydra search path:

```bash
# Single algorithm run
python -m manylatents.omics.main --config-name=config \
  experiment=single_algorithm data=pbmc_3k

# Sweep on cluster
python -m manylatents.omics.main -m \
  cluster=tamia resources=gpu \
  data=hgdp,pbmc_10k algorithms/latent=umap,phate
```

Never use `manylatents.main` for omics data — it won't find omics configs.

## Foundation model encoders

| Encoder | Domain | VRAM | Embedding dim |
|---------|--------|------|---------------|
| ESM3 | Protein | 16 GB+ | 1536 |
| Evo2 | DNA | 24 GB+ (Ampere+) | 2048 |
| Orthrus | RNA | 8 GB+ | 256 / 512 |
| AlphaGenome | DNA | 40 GB+ (JAX) | 1536 / 3072 |

Ampere+ GPUs required: A100, L40S, H100, H200. Older GPUs (RTX 8000, V100) will fail.

## Datasets

**Population genetics** — HGDP+1KGP, UK Biobank, All of Us. Supports QC filtering, relatedness filtering, admixture proportions, geographic metadata.

**Single-cell** — AnnData format. Ships with PBMC 3k/10k/68k and Embryoid Body configs. Any `.h5ad` works via `AnnDataModule`.

**ClinVar** — DNA/protein variant encoding pipeline:

```bash
python -m manylatents.omics.main experiment=clinvar/encode_dna
python -m manylatents.omics.main experiment=clinvar/encode_protein
python -m manylatents.omics.main experiment=clinvar/geometric_analysis
```

## Metrics

- **GeographicPreservation** — embedding vs. geographic distance correlation
- **AdmixturePreservation** — ancestry proportion fidelity in latent space
- K-curve analysis across neighborhood sizes

## Development

```bash
uv sync --extra dev
pytest tests/ -v
```

---

<div align="center">

MIT License · Latent Reasoning Works

</div>
