# Release v0.2.0 Roadmap

## Summary

Ship manylatents-omics v0.2.0 to PyPI. The package code and CI/CD pipeline are ready — what remains is merge hygiene, test guard fixes, basic singlecell test coverage, and cleanup of development artifacts.

## Motivation

The package has been at v0.2.0 in `pyproject.toml` since 2026-02-18 but has never been published. All three modules (popgen, singlecell, dogma) are functional, CI runs on Python 3.11/3.12, and the PyPI publish workflow is configured via OIDC trusted publishing. Five open issues all have PRs ready to merge.

## Scope

This change covers the work between current `main` and tagging `v0.2.0`. It does **not** include new features — only merge, fix, test, and clean tasks.

## Current State (as of 2026-03-03)

### What's ready
- Package metadata, build system (hatchling), extras all correct
- popgen: 27/27 tests passing, fully covered
- dogma: architecture solid, all non-GPU tests pass
- CI/CD: test matrix (3.11/3.12), docs deploy, PyPI publish workflow
- Documentation: mkdocs-material with comprehensive module docs
- License (MIT) + CITATION.cff

### What's blocking
- 5 open issues with PRs pending merge
- 12 test failures from missing `pytest.importorskip` guards
- PR #20 has merge conflicts
- Stale PR #5 needs triage
- Singlecell module has 0 tests
- Root-level development debris (`.out` files, ad-hoc scripts)
- No CHANGELOG.md
