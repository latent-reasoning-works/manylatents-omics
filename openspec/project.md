# Manylatents-Omics - Project Context

## Purpose
Extension package for manylatents providing:
- Foundation model encoders (ESM3, Evo2, Orthrus)
- Central dogma fusion (DNA + RNA + Protein embeddings)
- Population genetics datasets (HGDP, AOU, UKBB)
- Single-cell omics datasets (PBMC, Embryoid Body)

## Tech Stack
- Python 3.10+
- Inherits from manylatents core
- Shop for cluster submission (optional)
- Hydra SearchPathPlugin for config discovery
- **Wheelnext uv** for dogma extras (prebuilt CUDA wheels)

## Project Conventions

### Module Structure
| Module | Domain | Data Format |
|--------|--------|-------------|
| manylatents.dogma | DNA/RNA/Protein | FASTA/sequences |
| manylatents.popgen | Population genetics | PLINK |
| manylatents.singlecell | Single-cell | AnnData |

### Hydra Config Pattern
- Configs in `manylatents/<module>/configs/`
- SearchPathPlugin in `hydra_plugins/manylatents_omics/`
- `# @package _global_` MUST be on line 1 of experiment configs

## Important Constraints
- Model weights on cluster at /network/weights/
- manylatents.main needs `--config-name=config`
- GPU requirements: 24GB+ VRAM for all 3 encoders, Ampere+ for FP8
- **Dogma installation requires wheelnext uv** (see CLAUDE.md)
  - transformer-engine-torch, mamba-ssm have no PyPI wheels
  - Uses `--index-strategy unsafe-best-match` for cross-index resolution
  - Login nodes need `module load cuda/12.4.1` for imports

## Dogma Installation Quick Reference
```bash
# 1. Install wheelnext uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh

# 2. Lock and sync
uv lock --index-strategy unsafe-best-match
uv sync --extra dogma --index-strategy unsafe-best-match

# 3. Verify (on login node)
module load cuda/12.4.1
uv run python -c "import evo2, orthrus, esm"
```

## External Dependencies
- manylatents (core dimensionality reduction)
- shop (optional, for SLURM launchers)
- wheelnext uv (for dogma CUDA packages): https://astral.sh/blog/wheel-variants

## Version Pins (ABI Compatibility)
| Package | Constraint | Reason |
|---------|------------|--------|
| torch | >=2.7,<2.8 | transformer-engine-torch ABI |
| torchvision | >=0.22,<0.23 | Must match torch |
| nvidia-cudnn-cu12 | >=9.5 | Runtime libraries |
