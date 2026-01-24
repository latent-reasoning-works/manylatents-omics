# manylatents-omics Installation & Usage Guide

This guide covers installing and using manylatents-omics, the biological extensions package for manylatents.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/latent-reasoning-works/manylatents-omics.git
cd manylatents-omics

# Install dependencies (pulls manylatents from git automatically)
uv sync

# Run an experiment
uv run python -m manylatents.omics.main --config-name=config experiment=single_algorithm
```

## Prerequisites

### 1. UV Package Manager

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. CUDA Environment (for GPU features)

On Mila cluster, load CUDA modules before running:

```bash
module load anaconda/3 cuda/12.4.1
```

On other systems, ensure CUDA 12.x is installed and `libcufile.so` is available.

## Installation Options

### Option A: Development Install (Recommended)

Best for contributing to manylatents-omics:

```bash
git clone https://github.com/latent-reasoning-works/manylatents-omics.git
cd manylatents-omics
uv sync
```

### Option B: Install as Dependency

From another project that needs omics features:

```bash
# From your project directory
uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
```

**Note**: Editable installs (`uv add -e`) from another project may have Hydra plugin discovery issues. Use the git URL install instead.

### Option C: With Foundation Model Encoders (dogma)

For DNA/RNA/Protein encoding with Evo2, ESM3, Orthrus:

```bash
# Install wheelnext uv for CUDA wheel support
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh

# Sync with dogma extras
uv sync --extra dogma
```

## Running Experiments

### Entry Point

Always use the omics-specific entry point to ensure configs are discovered:

```bash
python -m manylatents.omics.main --config-name=config <overrides>
```

**Why not `python -m manylatents.main`?**

The omics entry point registers the Hydra SearchPathPlugin before initialization, making dogma/popgen/singlecell configs available. The standard manylatents entry point doesn't know about omics configs.

### Example Commands

```bash
# Core experiment with omics data
python -m manylatents.omics.main --config-name=config experiment=single_algorithm

# Dogma fusion experiment (requires GPU + dogma extras)
python -m manylatents.omics.main --config-name=config experiment=central_dogma_fusion

# ClinVar encoding (requires GPU)
python -m manylatents.omics.main --config-name=config experiment=clinvar/encode_dna

# View available configs
python -m manylatents.omics.main --help
```

### Config Groups

After installation, these config groups become available:

| Group | Description |
|-------|-------------|
| `dogma/configs/data/*` | Sequence datasets (ClinVar, DNA/RNA/Protein) |
| `dogma/configs/algorithms/latent/*` | Foundation model encoders (Evo2, ESM3, Orthrus) |
| `dogma/configs/experiment/*` | Pre-configured experiments |
| `dogma/configs/encoders/*` | Encoder configurations |

## Development Workflows

### Working on manylatents-omics

```bash
cd manylatents-omics
uv sync
uv run pytest  # Run tests
uv run python -m manylatents.omics.main --config-name=config experiment=single_algorithm
```

### Working on manylatents core with omics testing

```bash
cd manylatents
uv sync
# Install omics from git (NOT editable)
uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git
```

## Troubleshooting

### ImportError: libcufile.so.0

**Problem**: CUDA libraries not found.

**Solution**: Load CUDA modules:
```bash
module load anaconda/3 cuda/12.4.1
```

### ConfigAttributeError: Key 'experiment' is not in struct

**Problem**: Hydra SearchPathPlugin not registered.

**Solution**: Use the omics entry point:
```bash
# Wrong
python -m manylatents.main experiment=central_dogma_fusion

# Correct
python -m manylatents.omics.main --config-name=config experiment=central_dogma_fusion
```

### Could not override 'experiment'. No match in the defaults list

**Problem**: Missing `--config-name=config`.

**Solution**: Always specify the config name:
```bash
python -m manylatents.omics.main --config-name=config experiment=single_algorithm
```

### Verify Plugin Registration

Check that the omics plugin is registered:

```python
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

plugins = list(Plugins.instance().discover(SearchPathPlugin))
print([p.__name__ for p in plugins])
# Should include: OmicsSearchPathPlugin
```

## Architecture

### Hydra Config Discovery

manylatents-omics uses a custom entry point (`manylatents.omics.main`) that:

1. Registers `OmicsSearchPathPlugin` with Hydra
2. Adds `pkg://manylatents.dogma.configs` to the search path
3. Adds `pkg://manylatents.configs` (core) to the search path
4. Calls the standard `manylatents.main` function

This ensures omics configs are available regardless of installation method.

### Package Structure

```
manylatents-omics/
├── manylatents/
│   ├── omics/
│   │   ├── __init__.py
│   │   └── main.py          # Omics entry point
│   ├── omics_plugin.py      # Hydra SearchPathPlugin
│   ├── dogma/               # Foundation model encoders
│   │   ├── configs/         # Hydra configs
│   │   ├── encoders/        # Evo2, ESM3, Orthrus
│   │   ├── data/            # Sequence datasets
│   │   └── algorithms/      # Fusion algorithms
│   ├── popgen/              # Population genetics
│   └── singlecell/          # Single-cell omics
└── docs/
    └── README.md            # This file
```

## See Also

- [manylatents documentation](https://github.com/latent-reasoning-works/manylatents)
- [Hydra documentation](https://hydra.cc/)
- [Central Dogma Fusion](../CLAUDE.md#central-dogma-fusion-dna--rna--protein)
