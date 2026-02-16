# Manylatents-Omics

Extension of manylatents for biological data. Adds popgen, single-cell, and foundation model encoders.

**Inherits from**: `manylatents/CLAUDE.md` (architecture, contracts, safety rules)
**Infrastructure**: `shop/CLAUDE.md` (clusters, GPUs, log sync)

## Entry point

Omics configs are auto-discovered when the package is installed:

```bash
python -m manylatents.main experiment=single_algorithm data=pbmc_3k
```

## Three modules

| Module | Domain | Data format | Example |
|--------|--------|-------------|---------|
| `manylatents.popgen` | Population genetics | manifold-genetics CSVs | `data=hgdp` |
| `manylatents.singlecell` | Single-cell omics | AnnData `.h5ad` | `data=pbmc_3k` |
| `manylatents.dogma` | DNA/RNA/Protein | FASTA, sequences | `encoder=esm3` |

## Foundation model encoders

| Encoder | Domain | VRAM | Embedding dim |
|---------|--------|------|---------------|
| ESM3 | Protein | 16GB+ | 1536 |
| Evo2 | DNA | 24GB+ (Ampere+) | 2048 |
| Orthrus | RNA | 8GB+ | 256/512 |
| AlphaGenome | DNA | 40GB+ (JAX) | 1536/3072 |

## Install

```bash
# Base (popgen + singlecell)
uv sync

# Foundation models (requires wheelnext uv)
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
uv lock --index-strategy unsafe-best-match
uv sync --extra dogma --index-strategy unsafe-best-match

# On login nodes, load CUDA before imports
module load cuda/12.4.1
```

## Fire experiments

```bash
# Local test
python -m manylatents.main data=pbmc_3k algorithms/latent=umap

# Cluster submission (via shop configs)
python -m manylatents.main -m \
  cluster=tamia resources=gpu \
  experiment=clinvar/encode_dna

# Sweep
python -m manylatents.main -m \
  cluster=mila resources=gpu \
  data=hgdp,pbmc_10k \
  algorithms/latent=umap,phate
```

## Key files

| What | Where |
|------|-------|
| SearchPath plugin | `manylatents/omics_plugin.py` |
| Encoders | `manylatents/dogma/encoders/` |
| ClinVar data | `manylatents/dogma/data/` |
| PopGen data | `manylatents/popgen/data/` |
| SingleCell data | `manylatents/singlecell/data/` |
| Omics configs | `manylatents/dogma/configs/`, `manylatents/popgen/configs/`, `manylatents/singlecell/configs/` |

## ClinVar pipeline

```bash
# Step 1: Encode DNA
python -m manylatents.main experiment=clinvar/encode_dna

# Step 2: Encode protein
python -m manylatents.main experiment=clinvar/encode_protein

# Step 3: Fuse + geometric analysis
python -m manylatents.main experiment=clinvar/geometric_analysis
```

## Offline clusters (Tamia, Narval)

- `WANDB_MODE=offline` is set automatically by shop cluster configs
- After job completes, sync logs: `python -m shop.slurm.log_sync --cluster tamia --job-id <ID> --sync-wandb`
- Models must be cached locally — no HuggingFace downloads on compute nodes

## Commands

```bash
pytest tests/ -v
uv run python -c "import manylatents.popgen; import manylatents.singlecell; print('OK')"
uv run python -c "import manylatents.dogma; print('dogma OK')"  # needs CUDA
```
