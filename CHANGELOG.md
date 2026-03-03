# Changelog

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
