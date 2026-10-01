# Changelog

## Unreleased

### Added
- Local track-scoring CLI with required reference FASTA, contiguous sharding and reference/window prefiltering.
- Multi-directory Atlas cache loading with track metadata and duplicate-variant checks.
- Full TraitGym pool loading and reproducible uniform or consequence-matched background sampling with explicit shortfalls.
- NumPy/pandas benchmark helpers for genomic separation, chained loci, positional scores and distant neighbours.
- `manylatents.dogma.atlas`: resumable AlphaGenome Atlas score cache and variant-by-track matrix reader (`noncoding` extra). `LocalScorerClient` fills the same cache from a locally loaded AlphaGenome model.
- `manylatents.dogma.data.traitgym`: TraitGym benchmark tables.
- `manylatents.dogma.variants`: canonical variant identifiers.

## v0.1.1 (2026-03-03)

### Removed
- `experiments/` directory — prescriptive experiment configs, cluster presets, and ClinVar pipeline scripts moved to downstream experiment repos
- `AlphaGenomePredictor` (dead code, never exported)
- ClinVar-specific scripts from `scripts/` (deploy.sh, smoke-test.sbatch moved to shop)

### Fixed
- `docs/plans/` added to `.gitignore` (internal design docs excluded from package)

## v0.1.0 (2026-03-03)

First public release.

### Added
- **popgen** module: `ManifoldGeneticsDataset`, `ManifoldGeneticsDataModule`, `GeographicPreservation`, `AdmixturePreservation` metrics, admixture/geographic plot callbacks
- **singlecell** module: `AnnDataset`, `AnnDataModule` for `.h5ad` files, embedding audit pipeline (differential expression + complement set analysis)
- **dogma** module: `ESM3Encoder` (protein), `Evo2Encoder` (DNA), `OrthrusEncoder` (RNA), `AlphaGenomeEncoder` (DNA/JAX), `CentralDogmaFusion`, `LearnedFusion`
- ClinVar variant-effect analysis pipeline (encode DNA, encode protein, geometric analysis)
- Namespace extension architecture via `pkgutil.extend_path`
- Hydra SearchPath plugin for auto-discovery of omics configs
- CI/CD: tests on Python 3.11/3.12, docs deploy to GitHub Pages, PyPI publishing via OIDC

### Removed
- `AdmixtureLaplacian` metric (unused dead code)
- `AdmixturePreservationK` (consolidated into `AdmixturePreservation` with `admixture_k=None` for all-K mode)
- Empty `manylatents/omics/` scaffolding directory
