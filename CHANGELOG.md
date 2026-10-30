# Changelog

## Unreleased

- Relicensed from MIT to Apache-2.0. Releases up to and including 0.1.3 stay under MIT; NOTICE keeps the MIT notice.


### Added
- LARRY lineage-tracing dataset (Weinreb et al. 2020, doi:10.1126/science.aaw3381): `data=larry` config and `manylatents.singlecell.data.larry` (`LarryDataModule`, `fetch_larry`, `load_larry`), which download the public in-vitro files on first use and build `single_cell/larry.h5ad` with `Time point`, `clone`, `fate` and a 50-component PCA.
- `manylatents.singlecell.sampling.clone_subsample`: seeded whole-clone subsampling of lineage-traced cells up to a cell budget, keeping clones with an early cell and enough late descendants.
- `lookup_tabix_scores`: per-variant values from tabix-indexed score tables, with header/value-count validation.
- `leave_chromosome_out_probe`: nested chromosome-group tuning, out-of-fold decisions, parallel outer fits and explicit skipped-fold/row counts.
- `clustered_average_precision_difference`: paired, class-stratified cluster bootstrap intervals for pooled average precision differences.
- `genomic_blocks` and vectorised `nearest_site_distance` helpers, with explicit errors for query chromosomes lacking sites.
- Local track-scoring CLI annotation paths: `--gtf-feather`, `--pas-feather`, `--splice-starts-feather` and `--splice-ends-feather` for gene-scoped scorers.
- Chromosome splits matching AVI training and tuning partitions, with consistent chromosome prefixes.
- Pinned ncVarDB loader for pathogenic and benign nuclear SNVs, with exclusion counts.
- Full track matrices across cached scorers, including gene-scoped reductions and missing-row masks.
- Per-modality maximum and per-category reference-fitted PCA features, with omitted-track counts.
- Chromosome-split logistic probes with training-only standardisation, tuning selection and dropped-feature counts.
- Configurable track-geometry evaluation with permutation nulls, sensitivity grids, mismatched controls and locus-confound analysis.
- Local track-scoring CLI with required reference FASTA, contiguous sharding and reference/window prefiltering.
- Multi-directory Atlas cache loading with track metadata and duplicate-variant checks.
- Full TraitGym pool loading and reproducible uniform or consequence-matched background sampling with explicit shortfalls.
- NumPy/pandas benchmark helpers for genomic separation, chained loci, positional scores and distant neighbours.
- `manylatents.dogma.atlas`: resumable AlphaGenome Atlas score cache and variant-by-track matrix reader (`noncoding` extra). `LocalScorerClient` fills the same cache from a locally loaded AlphaGenome model.
- `manylatents.dogma.data.traitgym`: TraitGym benchmark tables.
- `manylatents.dogma.variants`: canonical variant identifiers.

### Changed
- The `noncoding` extra now requires `pysam>=0.22`; the dependency lock is refreshed.
- Split probes search C from 1e-6 to 1.0 by default and expose `chosen_c_on_edge`.

### Fixed
- Category PCA caps components at the standardised reference numerical rank and reports requested components withheld per scorer.
- Feature reductions preserve tuple unpacking and dropped-count metadata through pickle and copying.
- Probes drop and count numerically constant training features and accept validated object-dtype boolean labels.
- Chromosome splits normalise whitespace, case and integral numeric names and reject invalid or missing chromosomes.
- Track matrices zero-fill only attempted gene-scorer variants; incomplete caches raise `AtlasScoresMissing` listing unattempted IDs.
- Tabix lookups reject mismatched value names and fields instead of truncating rows silently.

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
