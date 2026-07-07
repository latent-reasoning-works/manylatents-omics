"""
Tests for the dataset loading path: malformed-dataset rejection and real 10x loads.

Demonstrates that:
1. Malformed datasets are refused at ingestion by the loader (``read_tenx`` /
   ``from_anndata``) — the core "reject bad data" guarantee. Runs offline against
   fabricated AnnData.
2. Real 10x datasets sampled from the dataset manifest load, validate, and have
   their dimension requirements enforced by ``require``.

The real-data tests are marked ``network``/``slow`` and skip gracefully when the
dataset manifest is absent or the host is offline. Knobs (all optional):
  - ``GEOMANCER_DATASETS_CSV``  path to the manifest CSV (overrides the in-repo default)
  - ``GEOMANCER_10X_CACHE``     directory to cache downloaded ``.h5`` files
  - ``GEOMANCER_MAX_MB``        skip datasets larger than this (default 1500)
  - ``GEOMANCER_TEST_SEED``     seed the random dataset pick for reproducibility
"""

import os
import re
import shutil
import socket
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from manykinds import LabeledArray
from manylatents.singlecell.data.manifests import select_random_tenx


# ==============================================================================
# Malformed-dataset rejection — bad data is refused at ingestion, not passed on.
# This is the core guarantee: a broken dataset must fail loudly at load, so it
# never reaches the model ops. Runs offline against fabricated AnnData.
# ==============================================================================


def _fabricated_adata(
    *,
    n_cells: int = 4,
    n_genes: int = 3,
    gene_ids: bool = True,
    feature_types: str | None = "Gene Expression",
    genome: bool = True,
    barcodes: list[str] | None = None,
):
    """A minimal in-memory AnnData mimicking a 10x read, mutated per case."""
    import anndata as ad
    import pandas as pd

    obs_index = barcodes if barcodes is not None else [f"AAAC{i}-1" for i in range(n_cells)]
    var = pd.DataFrame(index=[f"GENE{j}" for j in range(n_genes)])
    if gene_ids:
        var["gene_ids"] = [f"ENSG{j}" for j in range(n_genes)]
    if feature_types is not None:
        var["feature_types"] = [feature_types] * n_genes
    if genome:
        var["genome"] = ["GRCh38"] * n_genes
    X = (np.arange(n_cells * n_genes, dtype=float) + 1).reshape(n_cells, n_genes)
    return ad.AnnData(X=X, obs=pd.DataFrame(index=obs_index), var=var)


def _load_via_read_tenx(monkeypatch, adata, **kwargs):
    """Run a fabricated AnnData through the real ``read_tenx`` validation path."""
    from manylatents.singlecell.data.adapters.sources import tenx

    monkeypatch.setattr(tenx.sc, "read_10x_h5", lambda _path: adata)
    return tenx.read_tenx("ignored.h5", metadata={}, **kwargs)


class TestMalformedDatasetRejected:
    """Malformed datasets are refused by the loader (``read_tenx`` / ``from_anndata``)."""

    def test_missing_gene_ids_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="gene_ids"):
            _load_via_read_tenx(monkeypatch, _fabricated_adata(gene_ids=False))

    def test_non_scrna_modality_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="not scRNA-seq"):
            _load_via_read_tenx(monkeypatch, _fabricated_adata(feature_types="Antibody Capture"))

    def test_missing_genome_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="genome"):
            _load_via_read_tenx(monkeypatch, _fabricated_adata(genome=False))

    def test_empty_matrix_is_rejected(self, monkeypatch):
        # No genes -> no var_names -> degenerate/empty matrix, refused on read.
        with pytest.raises(ValueError, match="no var_names"):
            _load_via_read_tenx(monkeypatch, _fabricated_adata(n_genes=0))

    def test_missing_time_labels_is_rejected(self, monkeypatch):
        # use_time requested, but barcodes carry no '-<int>' time suffix.
        bad = _fabricated_adata(barcodes=["CellA", "CellB", "CellC", "CellD"])
        with pytest.raises(ValueError, match="time"):
            _load_via_read_tenx(monkeypatch, bad, use_time=True)

    def test_from_anndata_rejects_empty_directly(self):
        # The adapter itself refuses a zero-size matrix via LabeledArray.validate.
        from manylatents.singlecell.data.adapters.formats.adapters import from_anndata

        adata = _fabricated_adata(n_cells=3, n_genes=0)
        with pytest.raises(ValueError, match="empty"):
            from_anndata(adata, coords={"cell": list(adata.obs_names), "gene": []})


# ==============================================================================
# Success path — a well-formed 10x AnnData loads through the full read_tenx path.
# Guards the happy path offline (the manifest-backed real-data test is skipped
# when the manifest is absent), so a regression in the from_anndata handoff —
# e.g. an unexpected kwarg — is caught by CI rather than only in production.
# ==============================================================================


class TestValidDatasetLoads:
    """A valid fabricated AnnData loads to a LabeledArray via the real read_tenx."""

    def test_valid_adata_returns_labeled_array(self, monkeypatch):
        kind = _load_via_read_tenx(monkeypatch, _fabricated_adata())
        assert isinstance(kind, LabeledArray)
        assert {"cell", "gene"} <= set(kind.da.dims)
        assert kind.da.attrs["genome"] == "GRCh38"

    def test_use_time_extracts_time_coord(self, monkeypatch):
        # Default barcodes carry a trailing '-1', so use_time yields a time coord.
        kind = _load_via_read_tenx(monkeypatch, _fabricated_adata(), use_time=True)
        assert isinstance(kind, LabeledArray)
        assert "time" in kind.da.coords


# ==============================================================================
# Real 10x datasets sampled from the dataset manifest
# ==============================================================================


def _selected_datasets(n: int = 5):
    """Pick ``n`` datasets from the manifest registry.

    The registry (``manylatents.singlecell.data.manifests``) is the single source
    of truth; this test does not parse the spreadsheet itself. ``GEOMANCER_TEST_SEED``
    makes the pick reproducible. Raises ``FileNotFoundError`` if the manifest is
    absent; the caller catches that at collection time and turns it into a skip
    (the CSV is git-ignored, so it is legitimately missing on a fresh clone / CI).
    """
    seed = os.environ.get("GEOMANCER_TEST_SEED")
    seed = int(seed) if (seed and seed.isdigit()) else None
    return select_random_tenx(n, seed=seed)


# Resolved once at collection time so each dataset is a separate test case.
# A missing manifest is deferred to a single skipping case (below) rather than
# raised here, so it doesn't error-out collection of the offline tests too.
try:
    _SELECTED_DATASETS = _selected_datasets(5)
    _MANIFEST_ERROR: str | None = None
except FileNotFoundError as e:
    _SELECTED_DATASETS = []
    _MANIFEST_ERROR = str(e)


def _short_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:40]


def _cache_dir() -> Path:
    d = Path(os.environ.get("GEOMANCER_10X_CACHE") or (Path(tempfile.gettempdir()) / "geomancer_10x_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_h5(url: str) -> Path:
    """Download (and cache) a 10x ``.h5``; skip the test on network failure or size cap."""
    dest = _cache_dir() / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    max_mb = int(os.environ.get("GEOMANCER_MAX_MB", "1500"))
    req = urllib.request.Request(url, headers={"User-Agent": "geomancer-tests"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            length = resp.headers.get("Content-Length")
            if length and int(length) > max_mb * 1024 * 1024:
                pytest.skip(
                    f"{dest.name} is {int(length) // (1024 * 1024)} MB > cap {max_mb} MB "
                    f"(raise GEOMANCER_MAX_MB to include it)"
                )
            part = dest.with_suffix(dest.suffix + ".part")
            with open(part, "wb") as out:
                shutil.copyfileobj(resp, out)
        part.rename(dest)
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
        if dest.exists():
            dest.unlink()
        pytest.skip(f"could not download {url}: {e}")
    return dest


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    "entry",
    # ``[None]`` sentinel when the manifest is absent so the case still runs and
    # skips explicitly (see the guard below) instead of collapsing to an empty skip.
    _SELECTED_DATASETS or [None],
    ids=[_short_id(e.name) for e in _SELECTED_DATASETS] or ["manifest-missing"],
)
def test_random_10x_dataset_loads_validates_and_enforces_dims(entry):
    """A randomly chosen 10x dataset loads, validates, and respects op dim contracts."""
    from manylatents.singlecell.data.adapters.sources.tenx import read_tenx

    if entry is None:
        # The manifest CSV is git-ignored and absent on a fresh clone / CI, so we
        # can't see it there — skip rather than fail, letting the PR go green.
        pytest.skip(
            f"dataset manifest not found — skipping online 10x tests: {_MANIFEST_ERROR}"
        )

    h5 = _download_h5(entry.url)

    # 1. Loading: real 10x .h5 -> typed LabeledArray via the production adapter.
    #    The manifest's Use_Time column drives whether a per-cell time coord is
    #    extracted from the barcode suffix.
    try:
        kind = read_tenx(str(h5), use_time=entry.use_time)
    except ValueError as e:
        # The adapter rejects non-scRNA-seq (antibody/CRISPR) modalities at the
        # edge — that's correct behavior, just not testable as scRNA-seq here.
        if "not scRNA-seq" in str(e):
            pytest.skip(f"{entry.name} is multimodal, not pure scRNA-seq: {e}")
        raise
    assert isinstance(kind, LabeledArray)

    # 2. Validation: the loaded kind wraps a well-formed DataArray and satisfies
    #    the cell/gene contract every downstream op assumes.
    kind.validate()
    kind.require("cell", "gene", coords=("cell", "gene"))
    assert {"cell", "gene"} <= set(kind.da.dims)
    assert {"cell", "gene"} <= set(kind.da.coords)

    # 3. Requiring dimensions for specific ops.
    #    A temporal op needs a 'time' dim, which raw 10x data lacks, so require
    #    must reject this kind cleanly rather than fail deep in a computation.
    assert "time" not in kind.da.dims
    with pytest.raises(ValueError, match="requires dims"):
        kind.require("cell", "gene", "time")

    #    A cell/gene-only op is satisfied. Exercise it on a small dense slice so
    #    we don't materialize the full (often >10k×30k) sparse matrix.
    n_cells = min(64, kind.da.sizes["cell"])
    n_genes = min(128, kind.da.sizes["gene"])
    small = kind.da.isel(cell=slice(0, n_cells), gene=slice(0, n_genes))
    small = small.copy(data=np.asarray(small.data.todense()))
    small_kind = LabeledArray(small)

    small_kind.validate()
    small_kind.require("cell", "gene")
    assert {"cell", "gene"} <= set(small_kind.da.dims)
