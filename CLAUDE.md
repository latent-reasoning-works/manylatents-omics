# CLAUDE.md

Biological extensions for manylatents: popgen, single-cell, and foundation model encoders.

**See [ARCHITECTURE.md](ARCHITECTURE.md) for the codebase map, data flow, and entrypoint alignment issues.**

**Inherits from**: `manylatents/CLAUDE.md` (core contracts, safety rules)
**Infrastructure**: `shop/CLAUDE.md` (clusters, GPUs, log sync)

## Before Starting Work

- **Check ARCHITECTURE.md** for the entrypoint alignment notes before adding configs.
- **Follow core conventions** for new components. Metrics are flat YAML with `at:` field.
  LatentModules get `algorithms/latent/` configs. LightningModules get `algorithms/lightning/`.
- **Lazy imports for heavy deps.** Encoder files must import without CUDA installed.
  Put `import torch`, model loads, etc. inside `_load_model()` or method bodies.

## Entry Point

Omics configs are auto-discovered when the package is installed:

```bash
python -m manylatents.main data=pbmc_3k algorithms/latent=umap
```

## Install

```bash
# Base (popgen + singlecell)
uv sync

# Foundation models (requires wheelnext uv)
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
uv sync --extra dogma

# On login nodes, load CUDA before imports
module load cuda/12.4.1
```

## Commands

```bash
pytest tests/ -v
uv run python -c "import manylatents.popgen; import manylatents.singlecell; print('OK')"
uv run python -c "import manylatents.dogma; print('dogma OK')"  # needs CUDA
```

## Gotchas

- **Evo2 needs a separate venv** — torch version conflict with other encoders. Use `scripts/setup-dna-venv.sh`.
- **mamba-ssm must be exactly 2.2.6.post3** — Orthrus breaks above this version.
- **ESM3 must stay float32** — dtype mismatch bug in the ESM library.
- **Offline clusters (Tamia, Narval)** — models must be cached locally. No HuggingFace downloads on compute nodes. `WANDB_MODE=offline` is set automatically.
- **`${omics_data:}` resolver** — use in data configs for `.h5ad` paths. Resolves to `<repo>/data/`.

---

*Last updated: 2026-04-01*
