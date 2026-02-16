# manylatents-omics docs

## Two environments

RNA/Protein and DNA encoders need different torch versions. Two separate venvs solve this.

```
.venv/        ← RNA + Protein (torch 2.5.1 + cu121)
.venv-dna/    ← DNA           (torch 2.8.0 + cu126)
```

### Why two envs?

- **Orthrus** (RNA) depends on `mamba-ssm`, which pins `torch==2.5.1`
- **Evo2** (DNA) depends on `transformer-engine`, which requires `torch>=2.8`
- These two torch versions are mutually exclusive in a single venv

### RNA + Protein env (`.venv`)

This is the default env managed by `uv sync`:

```bash
uv sync --extra dogma
```

This installs torch 2.5.1 + mamba-ssm 2.2.6.post3 + ESM3 + Orthrus. The lockfile pins everything via the `pytorch-cu121` index.

**What runs here**: ESM3 (protein encoding), Orthrus (RNA encoding), all popgen/singlecell workloads.

**GPU requirement**: Any modern GPU (V100+). No Ampere restriction.

**SLURM resource config**: `resources=gpu_rna`
- Activates `.venv`, loads `cuda/12.1.1`
- Works on Mila (any GPU), Narval (A100), Cedar/Beluga (V100)

### DNA env (`.venv-dna`)

Built manually via the setup script:

```bash
source scripts/setup-dna-venv.sh
```

This creates `.venv-dna/` with torch 2.8.0+cu126, transformer-engine 2.11 (from NVIDIA GitHub release wheel), flash-attn (built from source), and evo2.

**What it does step by step**:
1. Creates `.venv-dna` with Python 3.12
2. Installs `torch==2.8.0` from `pytorch-cu126` index
3. Downloads transformer-engine wheel from NVIDIA GitHub, renames for uv compatibility
4. Installs `flash-attn` with `--no-build-isolation` (builds from source, needs CUDA)
5. Installs `evo2` and `esm>=3.0`
6. Installs manylatents + manylatents-omics

**What runs here**: Evo2 (DNA encoding), ESM3 (also works here), AlphaGenome.

**GPU requirement**: Ampere+ (A100, L40S, H100, H200). BF16 required.

**SLURM resource configs**:
- `resources=gpu_dna` — Mila, activates `.venv-dna`, loads `cuda/12.5.0`, requests A100
- `resources=gpu_tamia_dna` — Tamia, activates `.venv-dna`, loads `cuda/12.6`, requests H100x4

### Which env for which encoder?

| Encoder | Env | Torch | Resource config |
|---------|-----|-------|-----------------|
| ESM3 (protein) | `.venv` | 2.5.1 | `gpu_rna` |
| Orthrus (RNA) | `.venv` | 2.5.1 | `gpu_rna` |
| Evo2 (DNA) | `.venv-dna` | 2.8.0 | `gpu_dna` / `gpu_tamia_dna` |
| AlphaGenome (DNA) | `.venv-dna` | 2.8.0 | `gpu_dna` |

### Submitting jobs with the right env

The resource configs handle venv activation automatically. Just match the resource to the encoder:

```bash
# RNA/Protein — uses .venv
python -m manylatents.omics.main -m \
  experiment=clinvar/encode_protein \
  cluster=mila resources=gpu_rna

# DNA — uses .venv-dna
python -m manylatents.omics.main -m \
  experiment=clinvar/encode_dna \
  cluster=mila resources=gpu_dna

# DNA on Tamia — uses .venv-dna with H100
python -m manylatents.omics.main -m \
  experiment=clinvar/encode_dna \
  cluster=tamia_submitit resources=gpu_tamia_dna
```

### Offline clusters (Tamia, Narval)

Both venvs must be built on Mila (internet access), then transferred:

```bash
# On Mila: build both envs
uv sync --extra dogma                  # .venv
source scripts/setup-dna-venv.sh       # .venv-dna

# Transfer to Tamia
rsync -avz --exclude __pycache__ \
  .venv .venv-dna manylatents/ scripts/ pyproject.toml \
  tamia:/scratch/c/user/project/

# On Tamia: fix Python paths
bash scripts/setup-tamia.sh
```

The Tamia setup script patches `pyvenv.cfg` to point at Tamia's Python and reinstalls manylatents + omics in editable mode (no network needed).

## CUDA modules

| Cluster | RNA/Protein env | DNA env |
|---------|----------------|---------|
| Mila | `cuda/12.1.1` | `cuda/12.5.0` |
| Tamia | `cuda/12.6` | `cuda/12.6` |

## Entry point

Always use the omics entry point:

```bash
python -m manylatents.omics.main --config-name=config <overrides>
```

This registers the Hydra SearchPathPlugin that makes dogma/popgen/singlecell configs discoverable. The standard `manylatents.main` won't find omics configs.

## Troubleshooting

**`ImportError: libcufile.so.0`** — Load CUDA module: `module load cuda/12.1.1`

**`ConfigAttributeError: Key 'experiment' is not in struct`** — Use `python -m manylatents.omics.main`, not `manylatents.main`

**`Could not override 'experiment'`** — Add `--config-name=config`

**mamba-ssm build fails** — The lockfile uses `no-build-isolation-package` so mamba-ssm sees the venv's torch 2.5.1 and downloads a prebuilt wheel. If this fails, ensure `uv sync` (not `uv pip install`) is used for the RNA env.

**transformer-engine won't install** — Use the DNA setup script, not `uv sync`. The wheel comes from NVIDIA GitHub with a non-standard version string that uv can't resolve from PyPI.
