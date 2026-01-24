# Design: Consolidate Hydra Config Infrastructure

## Architecture Overview

### Config Discovery Flow
```
manylatents.main (entry point)
    ↓
hydra_plugins/manylatents_omics/searchpath.py
    ↓
discovers: manylatents/dogma/configs/
    ├── encoders/   (esm3, evo2, orthrus)
    ├── data/       (sequences_*)
    ├── experiment/ (central_dogma_fusion)
    └── logger/     (wandb_shop) ← NEW
```

### Logger Config Integration

The `wandb_shop.yaml` config uses `wandb.init` directly with `config: null`
to allow programmatic override by experiment.py:

```yaml
# wandb_shop.yaml
# @package _global_
logger:
  _target_: wandb.init
  project: ${project}
  name: ${name}
  config: null  # Set programmatically by experiment.py
  mode: online
```

### Dependency Structure

```
manylatents-omics
├── [required]    manylatents, httpx, pandas-plink
├── [singlecell]  anndata, scanpy, igraph
├── [dogma]       torch, esm, evo2, orthrus, mamba-ssm, flash-attn, transformer-engine
├── [cluster]     shop  ← optional for SLURM submission
└── [dev]         pytest, ruff
```

## Key Design Decisions

1. **Shop as optional**: Users doing local-only work shouldn't need submitit/shop
2. **Logger in dogma/configs**: Keeps omics-specific configs self-contained
3. **No eval_only**: Central dogma fusion runs encoders, doesn't load precomputed
4. **`_recursive_: false`**: Prevents Hydra from instantiating encoders prematurely

## CUDA Package Installation Strategy

### Problem
The dogma extras require GPU-accelerated packages that have no prebuilt wheels on PyPI:

| Package | PyPI Status | Solution |
|---------|-------------|----------|
| transformer-engine-torch | No wheels | Wheelnext uv |
| mamba-ssm | Source only | Wheelnext uv |
| flash-attn | Needs CUDA_HOME | Wheelnext uv |

### Solution: Wheelnext UV

Wheelnext is an experimental uv build supporting [wheel variants](https://astral.sh/blog/wheel-variants) -
a proposed PEP for encoding GPU/CUDA variants in package metadata.

```bash
# Install wheelnext uv (replaces standard uv)
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
```

### Version Compatibility Matrix

| Package | Version | Constraint Reason |
|---------|---------|-------------------|
| torch | 2.7.x | ABI match with transformer-engine-torch |
| torchvision | 0.22.x | Must match torch major version |
| transformer-engine-torch | 2.11.x | Built against torch 2.7 |
| mamba-ssm | 2.3.x | Wheelnext prebuilt wheel |
| flash-attn | 2.8.x | Wheelnext prebuilt wheel |

### Cross-Index Resolution

The wheelnext PyTorch index has newer torch versions than we need. To resolve
torch 2.7.x from PyPI while getting CUDA packages from wheelnext:

```bash
uv lock --index-strategy unsafe-best-match
uv sync --extra dogma --index-strategy unsafe-best-match
```

### Orthrus Compatibility Shim

mamba-ssm 2.x moved the `Block` class from `mamba_ssm.modules.mamba_simple` to
`mamba_ssm.modules.block`. Added shim in `orthrus.py`:

```python
try:
    from mamba_ssm.modules.mamba_simple import Block
except ImportError:
    from mamba_ssm.modules.block import Block
    import mamba_ssm.modules.mamba_simple as mamba_simple
    mamba_simple.Block = Block
```

## Runtime Requirements

### Login Node (for imports/testing)
```bash
module load cuda/12.4.1
uv run python -c "import evo2, orthrus, esm"
```

### GPU Node (via SLURM)
CUDA libraries are automatically available - no module load needed.

## Verified Package Versions

After successful `uv sync --extra dogma`:
```
torch==2.7.1+cu126
torchvision==0.22.1+cu126
transformer_engine==2.11.0
mamba_ssm==2.3.0
flash_attn==2.8.3
esm==3.1.3
evo2==1.0.0
orthrus==0.1.0
```
