"""
Tests for typed kinds: construction, round-trip, validation, and op contracts.

Demonstrates that:
1. A kind survives the construct → serialize → load → validate cycle.
2. Malformed kinds are rejected by ``validate`` (missing dims/coords, null coords).
3. Required dimensions are enforced structurally by ops.
4. Real 10x datasets (sampled from the Geomancer CSV) load, validate, and have
   their dimension requirements enforced by ops.

The real-data tests are marked ``network``/``slow`` and skip gracefully when the
dataset CSV is absent or the host is offline. Knobs (all optional):
  - ``GEOMANCER_DATASETS_CSV``  path to the datasets CSV
  - ``GEOMANCER_10X_CACHE``     directory to cache downloaded ``.h5`` files
  - ``GEOMANCER_MAX_MB``        skip datasets larger than this (default 1500)
  - ``GEOMANCER_TEST_SEED``     seed the random dataset pick for reproducibility
"""

import csv
import os
import random
import re
import shutil
import socket
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from manylatents.singlecell.data.kinds.kinds import Kind, LabeledArray


# ==============================================================================
# Base Kind
# ==============================================================================


def test_kind_is_abstract():
    """The abstract base cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Kind()  # type: ignore[abstract]


# ==============================================================================
# LabeledArray: construct → serialize → load round-trip
# ==============================================================================


class TestLabeledArrayRoundTrip:
    """Test construct → serialize → load → validate cycle."""

    def test_round_trip_basic(self):
        """Round-trip with minimal data preserves values and structure."""
        data = np.random.rand(10, 5)
        da = xr.DataArray(
            data,
            dims=["cell", "gene"],
            coords={"cell": [f"c{i}" for i in range(10)], "gene": [f"g{i}" for i in range(5)]},
        )
        kind = LabeledArray(da)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.zarr")
            kind.serialize(path)
            loaded = LabeledArray.load(path)

            loaded.validate()
            assert (loaded.data == kind.data).all()
            assert list(loaded.data.dims) == list(kind.data.dims)
            assert list(loaded.data.cell) == list(kind.data.cell)
            assert list(loaded.data.gene) == list(kind.data.gene)

    def test_round_trip_with_attrs(self):
        """Round-trip preserves attributes."""
        da = xr.DataArray(
            np.ones((5, 3)),
            dims=["cell", "gene"],
            coords={"cell": ["c1", "c2", "c3", "c4", "c5"], "gene": ["g1", "g2", "g3"]},
            attrs={"genome": "GRCh38", "source": "10x"},
        )
        kind = LabeledArray(da)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.zarr")
            kind.serialize(path)
            loaded = LabeledArray.load(path)

            assert loaded.data.attrs["genome"] == "GRCh38"
            assert loaded.data.attrs["source"] == "10x"

    def test_round_trip_with_time_dim(self):
        """Round-trip with an optional time dimension."""
        da = xr.DataArray(
            np.random.rand(10, 5, 3),  # cells × genes × time
            dims=["cell", "gene", "time"],
            coords={
                "cell": [f"c{i}" for i in range(10)],
                "gene": [f"g{i}" for i in range(5)],
                "time": [0, 1, 2],
            },
        )
        kind = LabeledArray(da, required_dims={"cell", "gene", "time"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.zarr")
            kind.serialize(path)
            loaded = LabeledArray.load(path)

            assert "time" in loaded.data.dims
            assert len(loaded.data.time) == 3


# ==============================================================================
# LabeledArray: validation
# ==============================================================================


class TestLabeledArrayValidation:
    """Test structural enforcement of required dims and coords."""

    def test_validate_passes_with_required_dims(self):
        da = xr.DataArray(
            np.ones((3, 2)),
            dims=["cell", "gene"],
            coords={"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
        LabeledArray(da, required_dims={"cell", "gene"}).validate()  # no raise

    def test_validate_fails_missing_cell_dim(self):
        da = xr.DataArray(np.ones((5,)), dims=["gene"])
        kind = LabeledArray(da, required_dims={"cell", "gene"})
        with pytest.raises(ValueError, match="missing required dims"):
            kind.validate()

    def test_validate_fails_wrong_dims(self):
        da = xr.DataArray(
            np.ones((10, 5)),
            dims=["samples", "features"],  # wrong names
            coords={"samples": range(10), "features": range(5)},
        )
        kind = LabeledArray(da, required_dims={"cell", "gene"})
        with pytest.raises(ValueError, match="missing required dims"):
            kind.validate()

    def test_validate_fails_missing_required_coord(self):
        """A declared coordinate that is absent fails validation."""
        da = xr.DataArray(
            np.ones((3, 2)),
            dims=["cell", "gene"],
            coords={"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
        kind = LabeledArray(
            da, required_dims={"cell", "gene"}, required_coords={"cell", "gene", "gene_ids"}
        )
        with pytest.raises(ValueError, match="missing coordinates"):
            kind.validate()

    def test_validate_fails_on_null_coord(self):
        """Null values in a required coordinate (metadata loss) are rejected."""
        da = xr.DataArray(
            np.ones((3, 2)),
            dims=["cell", "gene"],
            coords={
                "cell": ["c1", "c2", "c3"],
                "gene": ["g1", "g2"],
                "gene_ids": ("gene", np.array(["ENSG1", None], dtype=object)),
            },
        )
        kind = LabeledArray(
            da, required_dims={"cell", "gene"}, required_coords={"gene_ids"}
        )
        with pytest.raises(ValueError, match="null values"):
            kind.validate()


# ==============================================================================
# Op contracts: requiring dimensions
# ==============================================================================


class TestOpDimensionRequirements:
    """Ops declare and enforce the dims they consume."""

    def test_temporal_op_requires_time_dim(self):
        from tests.singlecell.test_op.example_ops import temporal_analysis

        da_time = xr.DataArray(
            np.random.rand(10, 5, 3),
            dims=["cell", "gene", "time"],
            coords={
                "cell": [f"c{i}" for i in range(10)],
                "gene": [f"g{i}" for i in range(5)],
                "time": [0, 1, 2],
            },
        )
        result = temporal_analysis(LabeledArray(da_time, required_dims={"cell", "gene", "time"}))
        assert isinstance(result, LabeledArray)
        assert "time" in result.data.dims

    def test_temporal_op_rejects_missing_time_dim(self):
        from tests.singlecell.test_op.example_ops import temporal_analysis

        da_no_time = xr.DataArray(
            np.random.rand(10, 5),
            dims=["cell", "gene"],
            coords={"cell": [f"c{i}" for i in range(10)], "gene": [f"g{i}" for i in range(5)]},
        )
        with pytest.raises(ValueError, match="requires 'time' dimension"):
            temporal_analysis(LabeledArray(da_no_time))

    def test_basic_filter_works_without_time(self):
        from tests.singlecell.test_op.example_ops import basic_filter

        da = xr.DataArray(
            np.random.rand(10, 5),
            dims=["cell", "gene"],
            coords={"cell": [f"c{i}" for i in range(10)], "gene": [f"g{i}" for i in range(5)]},
        )
        result = basic_filter(LabeledArray(da, required_dims={"cell", "gene"}), min_expression=0.1)
        assert isinstance(result, LabeledArray)
        assert result.data.dims == ("cell", "gene")

    def test_require_dims_helper_rejects_missing_dim(self):
        from tests.singlecell.test_op.example_ops import require_dims

        da = xr.DataArray(np.ones((4, 2)), dims=["cell", "gene"])
        with pytest.raises(ValueError, match="requires dimension"):
            require_dims(LabeledArray(da), "cell", "gene", "time")


# ==============================================================================
# Real 10x datasets sampled from the Geomancer CSV
# ==============================================================================


def _datasets_csv() -> Path | None:
    """Locate the Geomancer datasets CSV (env override, else repo root)."""
    env = os.environ.get("GEOMANCER_DATASETS_CSV")
    if env:
        p = Path(env)
        return p if p.exists() else None
    # tests/test_kinds.py -> tests -> manylatents-omics -> lrw (repo root)
    candidate = Path(__file__).resolve().parents[2] / "Datasets for Geomancer - 10x Genomics.csv"
    return candidate if candidate.exists() else None


_H5_URL_RE = re.compile(r"https?://\S+?\.h5\b")


def _select_random_datasets(n: int = 5) -> list[tuple[str, str]]:
    """Pick ``n`` random (name, .h5 URL) pairs from the datasets CSV."""
    csv_path = _datasets_csv()
    if csv_path is None:
        return []

    rows: list[tuple[str, str]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            match = _H5_URL_RE.search(row.get("wget_commands") or "")
            if match:
                rows.append((row.get("Dataset_Name") or match.group(0), match.group(0)))
    if not rows:
        return []

    seed = os.environ.get("GEOMANCER_TEST_SEED")
    rng = random.Random(int(seed)) if (seed and seed.isdigit()) else random.Random()
    return rng.sample(rows, min(n, len(rows)))


# Resolved once at collection time so each dataset is a separate test case.
_SELECTED_DATASETS = _select_random_datasets(5)


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
@pytest.mark.skipif(not _SELECTED_DATASETS, reason="Geomancer 10x datasets CSV not found")
@pytest.mark.parametrize(
    "ds_name,url",
    _SELECTED_DATASETS,
    ids=[_short_id(name) for name, _ in _SELECTED_DATASETS],
)
def test_random_10x_dataset_loads_validates_and_enforces_dims(ds_name, url):
    """A randomly chosen 10x dataset loads, validates, and respects op dim contracts."""
    from manylatents.singlecell.data.adapters.sources.tenx import make_data
    from tests.singlecell.test_op.example_ops import basic_filter, temporal_analysis

    h5 = _download_h5(url)

    # 1. Loading: real 10x .h5 -> typed LabeledArray via the production adapter.
    try:
        kind = make_data(str(h5))
    except ValueError as e:
        # The adapter rejects non-scRNA-seq (antibody/CRISPR) modalities at the
        # edge — that's correct behavior, just not testable as scRNA-seq here.
        if "not scRNA-seq" in str(e):
            pytest.skip(f"{ds_name} is multimodal, not pure scRNA-seq: {e}")
        raise
    assert isinstance(kind, LabeledArray)

    # 2. Validation: the adapter-declared contract holds on the loaded kind.
    kind.validate()  # raises on missing dims/coords or null metadata
    assert {"cell", "gene"} <= set(kind.data.dims)
    assert {"cell", "gene"} <= set(kind.data.coords)
    assert kind.required_dims == {"cell", "gene"}

    # 3. Requiring dimensions for specific ops.
    #    temporal_analysis needs a 'time' dim, which raw 10x data lacks, so it
    #    must reject this kind cleanly rather than fail deep in a computation.
    assert "time" not in kind.data.dims
    with pytest.raises(ValueError, match="requires 'time' dimension"):
        temporal_analysis(kind)

    #    basic_filter only requires cell/gene. Exercise it on a small dense slice
    #    so we don't materialize the full (often >10k×30k) sparse matrix.
    n_cells = min(64, kind.data.sizes["cell"])
    n_genes = min(128, kind.data.sizes["gene"])
    small = kind.data.isel(cell=slice(0, n_cells), gene=slice(0, n_genes))
    small = small.copy(data=np.asarray(small.data.todense()))
    small_kind = LabeledArray(
        small, required_dims={"cell", "gene"}, required_coords={"cell", "gene"}
    )

    filtered = basic_filter(small_kind, min_expression=0.0)
    assert isinstance(filtered, LabeledArray)
    assert {"cell", "gene"} <= set(filtered.data.dims)
