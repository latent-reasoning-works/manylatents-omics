# Release Readiness

## ADDED Requirements

### Requirement: All open issues resolved before tagging

All GitHub issues MUST be closed (via merged PRs or manual close) before a version tag is created.

#### Scenario: Five issues closed by PR merge

Given issues #21, #22, #23, #24, #25 are open
When PRs #26, #27, #28, core#229 are merged
Then all five issues are auto-closed by GitHub
And `gh issue list --state open` returns empty

---

### Requirement: Test suite passes without failures

The test suite MUST have zero failures. Tests requiring optional GPU/model dependencies MUST skip cleanly via `pytest.importorskip`.

#### Scenario: Clean test run on CPU-only CI

Given the `dogma` extra is not installed
When `pytest tests/ -v` is executed
Then encoder encode tests are skipped (not failed)
And all other tests pass
And exit code is 0

---

### Requirement: Singlecell module has basic test coverage

The singlecell module MUST have at least import tests and basic AnnDataset/AnnDataModule tests using synthetic fixtures.

#### Scenario: Singlecell import and data loading

Given `manylatents-omics[singlecell]` is installed
When `pytest tests/singlecell/ -v` is executed
Then import tests pass
And AnnDataset loads a synthetic .h5ad fixture
And AnnDataModule sets up train/test dataloaders

---

### Requirement: Package builds and publishes cleanly

`uv build` MUST produce a valid wheel and sdist. The `publish.yml` workflow MUST trigger on `v*` tags and upload to PyPI.

#### Scenario: Build and tag-triggered publish

Given version in `pyproject.toml` is `0.2.0`
When `git tag v0.2.0 && git push origin v0.2.0` is executed
Then CI `publish.yml` triggers
And package is uploaded to PyPI as `manylatents-omics==0.2.0`
And `uv pip install manylatents-omics` succeeds

---

### Requirement: CHANGELOG documents the release

A `CHANGELOG.md` MUST exist at the repository root documenting what's in v0.2.0.

#### Scenario: CHANGELOG present with v0.2.0 entry

Given `CHANGELOG.md` exists at repository root
When the file is read
Then it contains a `## v0.2.0` section
And it lists added modules (popgen, singlecell, dogma)
And it lists removed items (AdmixtureLaplacian, AdmixturePreservationK)

---

### Requirement: No development debris in repository root

Ad-hoc test scripts, `.out` log files, and debug artifacts MUST be removed before release.

#### Scenario: Clean root directory

Given the repository root is checked
When `ls *.out *.log test_*.py inspect_*.py run_*.sh 2>/dev/null` is executed
Then no files are returned
