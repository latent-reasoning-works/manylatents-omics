# Release v0.2.0 — Tasks

## Phase 1: Merge open PRs (prerequisite for everything else)

- [ ] **1.1** Merge PR #26 — remove AdmixtureLaplacian, consolidate admixture metrics (closes #25)
- [ ] **1.2** Merge PR #27 — README restructure + manifold-genetics prerequisites (closes #23, #24)
- [ ] **1.3** Merge PR #28 — delete empty `manylatents/omics/` directory (closes #21)
- [ ] **1.4** Merge core repo PR #229 — wire omics extra in manylatents (closes #22)
- [ ] **1.5** Close or merge PR #5 ("Small Changes") — triage stale PR
- [ ] **1.6** Rebase PR #20 (embedding audit) onto updated main, resolve conflicts, merge

**Validation:** All 5 open issues closed. `gh issue list --state open` returns empty.

---

## Phase 2: Fix test suite

- [ ] **2.1** Add `pytest.importorskip` guards for optional encoder dependencies
  - `tests/dogma/test_encoders.py` — guard ESM3 encode tests with `pytest.importorskip("esm")`
  - `tests/dogma/test_encoders.py` — guard Orthrus encode tests with `pytest.importorskip("orthrus")`
  - `tests/dogma/test_encoders.py` — guard Evo2 encode tests with `pytest.importorskip("evo2")`
  - `tests/dogma/test_encoders.py` — guard central dogma consistency test
  - `tests/dogma/encoders/test_alphagenome.py` — guard GPU tests with `pytest.importorskip("jax")`
- [ ] **2.2** Add basic singlecell tests
  - `tests/singlecell/test_imports.py` — import smoke tests
  - `tests/singlecell/test_anndata_dataset.py` — AnnDataset with synthetic fixture
  - `tests/singlecell/test_anndata_datamodule.py` — AnnDataModule setup/dataloaders

**Validation:** `uv run pytest tests/ -v` — 0 failures, all encoder tests skip cleanly without GPU deps.

---

## Phase 3: Cleanup

- [ ] **3.1** Delete root-level development debris
  - `test_alphagenome.py`, `test_all_encoders.py`, `inspect_api.py`, `inspect_predict.py`
  - All `*.out` files (`test_alphagenome_*.out`, `inspect_*.out`, `test_all_encoders_*.out`)
  - `run_test.sh`, `run_inspect.sh`, `run_inspect_predict.sh`, `run_all_encoders.sh`
- [ ] **3.2** Add root-level `.out` and common debris to `.gitignore`
  ```
  *.out
  activations_debug.log
  ```
- [ ] **3.3** Create `CHANGELOG.md`
  ```markdown
  # Changelog

  ## v0.2.0 (2026-03-XX)

  First public release.

  ### Added
  - popgen module: ManifoldGeneticsDataset/DataModule, GeographicPreservation, AdmixturePreservation
  - singlecell module: AnnDataset, AnnDataModule for .h5ad files
  - dogma module: ESM3, Evo2, Orthrus, AlphaGenome foundation model encoders
  - ClinVar variant-effect analysis pipeline (encode DNA → encode protein → geometric analysis)
  - Embedding audit pipeline (DE + complement set analysis) for singlecell
  - Namespace extension architecture via pkgutil.extend_path
  - CI/CD: tests on Python 3.11/3.12, docs deploy, PyPI publishing

  ### Removed
  - AdmixtureLaplacian metric (dead code)
  - AdmixturePreservationK (consolidated into AdmixturePreservation)
  - Empty manylatents/omics/ scaffolding directory
  ```

**Validation:** `git status` shows no untracked `.out` files. CHANGELOG exists.

---

## Phase 4: Pre-release verification

- [ ] **4.1** Full test suite: `uv run pytest tests/ -v` — all pass or skip
- [ ] **4.2** Import smoke tests:
  ```bash
  uv run python -c "import manylatents.popgen; print('popgen OK')"
  uv run python -c "import manylatents.singlecell; print('singlecell OK')"
  uv run python -c "from manylatents.omics_plugin import OmicsSearchPathPlugin; print('plugin OK')"
  ```
- [ ] **4.3** Build check: `uv build` produces wheel and sdist
- [ ] **4.4** Docs build: `uv run mkdocs build --strict` — no warnings
- [ ] **4.5** CI green on main after all merges

---

## Phase 5: Tag and publish

- [ ] **5.1** Ensure `pyproject.toml` version is `0.2.0` and `CITATION.cff` matches
- [ ] **5.2** Commit CHANGELOG + any final cleanup
- [ ] **5.3** Tag: `git tag v0.2.0 && git push origin v0.2.0`
- [ ] **5.4** CI `publish.yml` triggers automatically — verify package appears on PyPI
- [ ] **5.5** Verify install from PyPI: `uv pip install manylatents-omics && python -c "import manylatents.popgen"`
- [ ] **5.6** Archive this openspec change: `openspec archive release-v020-roadmap`

---

## Dependencies

```
Phase 1 (merge PRs)
  └── Phase 2 (fix tests) — needs clean main
       └── Phase 3 (cleanup) — can parallelize with Phase 2
            └── Phase 4 (verification) — needs Phases 2+3 done
                 └── Phase 5 (tag + publish)
```

## Notes

- PR #20 (embedding audit) adds `leidenalg` to `singlecell` extra — this is fine since core manylatents already depends on it
- AlphaGenome `embeddings_128bp` NotImplementedError is acceptable for v0.2.0 — tracked, documented, niche use case
- No branch protection currently on main — consider enabling after release
