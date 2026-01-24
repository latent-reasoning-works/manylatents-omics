# Change: Consolidate Hydra Config Infrastructure

## Status: COMPLETED

## Why
Three repos (shop, manylatents, omics) have fragmented Hydra infrastructure causing:
- WandB logger config conflict (`config: null` vs explicit `config=` arg)
- `eval_only: true` failure (requires precomputed_path not provided by encoders)
- `# @package _global_` must be on line 1 (subtle YAML requirement)
- CUDA package installation complexity (no PyPI wheels for transformer-engine-torch, mamba-ssm)

## What Changed
- Added shop as optional `[cluster]` dependency
- Created `logger/wandb_shop.yaml` for WandB integration
- Fixed `central_dogma_fusion.yaml` experiment config with `_recursive_: false`
- Documented Hydra config patterns for extensions
- Established wheelnext uv as requirement for dogma extras
- Added version pins for torch/torchvision ABI compatibility

## Impact
- Affected specs: hydra-integration (new capability)
- Affected code:
  - `manylatents/dogma/configs/logger/wandb_shop.yaml` (new)
  - `manylatents/dogma/configs/experiment/central_dogma_fusion.yaml`
  - `manylatents/dogma/encoders/orthrus.py` (mamba-ssm 2.x compatibility shim)
  - `pyproject.toml` (version pins, installation docs)
  - `CLAUDE.md` (installation instructions)

## Key Findings

### CUDA Package Installation
Standard uv/pip cannot install dogma extras because:
1. **transformer-engine-torch**: No wheels on PyPI, only on conda-forge
2. **mamba-ssm**: No wheels on PyPI, only source distributions
3. **flash-attn**: Requires CUDA_HOME at build time

**Solution**: Wheelnext uv with wheel variants support auto-detects GPU and selects CUDA-compatible prebuilt wheels.

### Version Pin Requirements
- `torch>=2.7,<2.8`: ABI compatibility with transformer-engine-torch (2.8.0 causes undefined symbol errors)
- `torchvision>=0.22,<0.23`: Must match torch version
- `nvidia-cudnn-cu12>=9.5`: Ensures runtime libraries are included

### Cross-Index Resolution
The wheelnext PyTorch index only has torch 2.8.0, but we need 2.7.x from PyPI. Requires:
```bash
uv lock --index-strategy unsafe-best-match
uv sync --extra dogma --index-strategy unsafe-best-match
```

### Login Node vs GPU Node
- Login nodes require `module load cuda/12.4.1` for imports
- SLURM GPU nodes have CUDA automatically available
- Job 8535476 verified successful execution on L40S GPU

## Verification
- [x] Config dry-run test passed
- [x] All imports verified (evo2, orthrus, esm3, mamba_ssm, flash_attn, transformer_engine)
- [x] GPU execution on L40S (3242 MiB memory, ~15 seconds)
- [x] Reproducible via `uv sync --extra dogma`
