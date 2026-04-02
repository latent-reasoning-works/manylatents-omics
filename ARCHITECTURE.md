# Architecture Overview

Manylatents-omics extends [manylatents](https://github.com/latent-reasoning-works/manylatents)
with biological data modules, foundation model encoders, and domain-specific metrics.
It installs under the `manylatents.*` namespace — core never imports from omics.

Read this document once, then use symbol search for specifics.
See [core ARCHITECTURE.md](../manylatents/ARCHITECTURE.md) for the engine, metric system, and base classes.

## High-Level Data Flow

```
                        ┌─────────────────────────────┐
User ──► CLI            │  manylatents core engine     │
  python -m             │  main.py / api.py            │
  manylatents.main      │  experiment.py               │
                        └──────────┬──────────────────-┘
                                   │
             ┌─────────────────────┼────────────────────┐
             ▼                     ▼                     ▼
     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
     │   popgen     │     │  singlecell  │     │    dogma     │
     │              │     │              │     │              │
     │ CSV→Dataset  │     │ .h5ad→       │     │ Seq→Encoder  │
     │ Manifold-    │     │  AnnDataset  │     │ DNA/RNA/Prot │
     │ Genetics     │     │              │     │              │
     │              │     │ scVI (⚡)     │     │ Fusion/Batch │
     │ geo/admix    │     │ scGPT (fit)  │     │              │
     │ metrics      │     │              │     │ Evo2, ESM3,  │
     └──────────────┘     └──────────────┘     │ Orthrus,     │
                                               │ AlphaGenome  │
                                               └──────────────┘

⚡ = LightningModule (uses Trainer)
All others = LatentModule (fit/transform)
```

## Project Structure

```
manylatents/
├── __init__.py              # pkgutil.extend_path() — namespace package
├── omics_plugin.py          # Hydra SearchPathPlugin + ${omics_data:} resolver
│
├── popgen/                  # Population genetics
│   ├── data/                # ManifoldGeneticsDataset, ManifoldGeneticsDataModule
│   ├── metrics/             # GeographicPreservation, AdmixturePreservation
│   ├── callbacks/           # PlotAdmixture, PlotEmbeddings
│   ├── utils/               # Label mappings
│   └── configs/
│       ├── metrics/         # geographic_preservation, admixture_preservation, gt_preservation
│       └── callbacks/       # plot_admixture
│
├── singlecell/              # Single-cell omics
│   ├── data/                # AnnDataset, AnnDataModule, CellXGene
│   ├── algorithms/          # SCVIModule (LightningModule), ScGPTEncoder (LatentModule)
│   │   └── _scgpt_vendor/   # Vendored scGPT model (MIT, avoids torchtext dep)
│   ├── analysis/            # ComplementSet, DifferentialExpression, EmbeddingAudit
│   ├── sampling.py          # GeosketchSampling
│   └── configs/
│       ├── data/            # pbmc_3k, pbmc_10k, pbmc_68k, embryoid_body, etc.
│       ├── algorithms/
│       │   ├── latent/      # scgpt (LatentModule)
│       │   └── lightning/   # scvi (LightningModule)
│       └── metrics/sampling/   # geosketch
│
├── dogma/                   # Central dogma foundation encoders
│   ├── encoders/            # Evo2, ESM3, Orthrus, AlphaGenome
│   ├── data/                # SequenceDataset, ClinVar, CentralDogma, Variant
│   ├── algorithms/          # CentralDogmaFusion, BatchEncoder, AutoencoderFusion
│   └── configs/
│       ├── data/            # sequences_dna/rna/protein, clinvar, precomputed
│       └── algorithms/latent/  # evo2, esm3, orthrus, alphagenome, fusion, batch_encoder, merging

tests/                       # Per-domain: dogma/, popgen/, singlecell/
data/                        # .h5ad files, manifold-genetics CSVs
```

## Extension Mechanism

Omics plugs into core via two mechanisms:

1. **Namespace package** — `pkgutil.extend_path()` in `__init__.py` lets `manylatents.dogma`
   etc. coexist with core's `manylatents` package across separate repos/installs.

2. **Hydra SearchPathPlugin** — `OmicsSearchPathPlugin` in `omics_plugin.py` registers
   three config packages. Dogma is prepended (higher priority), popgen and singlecell
   appended. Plugin self-registers on import; no `hydra_plugins` namespace needed.

3. **OmegaConf resolver** — `${omics_data:}` resolves to `<omics_repo>/data/`, used in
   data configs for `.h5ad` file paths.

4. **Entry point** — `[project.entry-points."manylatents.extensions"]` in pyproject.toml.
   Core's `main.py:_discover_extensions()` finds and imports the plugin at startup.

## Core Components

### Foundation Encoders (dogma)

All inherit from `FoundationEncoder` (core's `algorithms/latent/foundation_encoder.py`).
Contract: lazy `_load_model()`, `encode(sequence) → Tensor`, `embed_batch()` with OOM retry,
`embedding_dim` property, `modality` property.

| Encoder | Modality | Model | Dims | Context | Notes |
|---------|----------|-------|------|---------|-------|
| Evo2Encoder | DNA | StripedHyena2 | 1920 (1B) | 1M bp | Multi-layer extraction |
| ESM3Encoder | Protein | ESM3 1.4B | 1536 | 2000 aa | Mean-pooled, float32 only |
| OrthrusEncoder | RNA | Mamba SSM | 256/512 | Full | Native mamba-ssm 2.x reimpl |
| AlphaGenomeEncoder | DNA | JAX-based | 1536/3072 | 1M bp | Regulatory track predictions |

### Domain Algorithms

| Algorithm | Base Class | Domain | Interface |
|-----------|-----------|--------|-----------|
| SCVIModule | LightningModule | singlecell | `encode(x)`, `training_step()` |
| ScGPTEncoder | LatentModule | singlecell | `fit()` (no-op), `transform()` |
| CentralDogmaFusion | LatentModule | dogma | Concatenates multi-modal embeddings |
| BatchEncoder | LatentModule | dogma | Single-modality batch encoding |
| AutoencoderFusion | LatentModule | dogma | Learned bottleneck compression |
| FrobeniusAEFusion | LatentModule | dogma | AE + Jacobian penalty (MBYL-style) |

### Data Modules

| DataModule | Domain | Input | Output |
|------------|--------|-------|--------|
| ManifoldGeneticsDataModule | popgen | CSVs (manifold-genetics pipeline) | (N, P) PCA + admixture |
| AnnDataModule | singlecell | .h5ad (AnnData) | (N, G) gene counts |
| SequenceDataModule | dogma | List of sequences | str + int-encoded |
| CentralDogmaDataModule | dogma | Aligned DNA/RNA/Protein | Multi-modal dict |
| ClinVarDataModule | dogma | ClinVar variants | wt/mut channels |
| VariantDataModule | dogma | Precomputed + variants | Unified variant |

### Domain Metrics

| Metric | Domain | What it measures | `at:` |
|--------|--------|------------------|-------|
| GeographicPreservation | popgen | Spearman corr(haversine, embedding dist) | dataset |
| AdmixturePreservation | popgen | Geodesic fidelity in admixture simplex | dataset |
| GroundTruthPreservation | core (shared) | Generic GT distance preservation | dataset |

## Entrypoint Alignment Notes

Remaining gaps between omics configs and core conventions:

| Issue | Current State | Core Convention |
|-------|--------------|-----------------|
| **No popgen data configs** | Popgen has no preset `data/*.yaml` configs | Other domains provide data presets (popgen data paths are user-specific CSVs from manifold-genetics) |

Previously fixed (2026-04-01):
- SCVIModule config moved from `algorithms/latent/` to `algorithms/lightning/`
- Duplicate `encoders/` config group removed; all encoder configs consolidated under `algorithms/latent/`
- Evo2 embedding dim corrected from 2048 to 1920 (1B model) in fusion config

## Architectural Invariants

- **Core never imports from omics.** Zero references to `manylatents.dogma`, `.popgen`,
  or `.singlecell` in the core package. Omics is fully optional.

- **Lazy imports for heavy deps.** Encoder files must import without CUDA/model deps
  installed. Heavy imports (torch, esm, evo2, jax) go inside `_load_model()`.

- **Dogma uses lazy `__getattr__`.** `dogma/__init__.py` defers submodule imports to
  avoid circular imports during Hydra's config scanning.

- **Namespace is shared, not forked.** Both core and omics ship `manylatents/__init__.py`
  with `extend_path()`. Both get installed; Python merges them.

- **Configs are the API contract.** Every algorithm, data module, and metric that users
  should reference needs a YAML config. Omics configs live in domain-specific
  `configs/` directories, discovered via the SearchPathPlugin.

## External Dependencies

| Extra | Key packages | Notes |
|-------|-------------|-------|
| (base) | manylatents | Only hard dependency |
| singlecell | anndata, scanpy, geosketch, leidenalg | |
| popgen | pandas-plink | |
| dogma | esm, orthrus, mamba-ssm==2.2.6.post3 | Evo2 needs separate venv (torch conflict) |
| gpu | faiss-gpu-cu12 | Optional GPU kNN |

mamba-ssm pin is load-bearing — Orthrus breaks above 2.2.6.post3.
Evo2 requires torch 2.5.x which conflicts with other encoders — use `scripts/setup-dna-venv.sh`.

## Deployment

**Install**: `uv sync` (base), `uv sync --extra singlecell`, `uv sync --extra dogma`
(needs wheelnext uv for CUDA wheels).

**Cluster**: Inherits core SLURM configs (`cluster=mila`, `cluster=tamia`).
Offline clusters (Tamia, Narval) need pre-cached model weights — no HuggingFace downloads.

**CI**: `pytest tests/ -v`. Markers: `@pytest.mark.gpu`, `@pytest.mark.slow`.

## Project Identification

| | |
|---|---|
| **Project** | manylatents-omics |
| **Version** | 0.1.1 |
| **Repository** | github.com/latent-reasoning-works/manylatents-omics |
| **License** | MIT |
| **Last updated** | 2026-04-01 |
