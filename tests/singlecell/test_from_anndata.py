"""Tests for the AnnData -> LabeledArray format adapter (``from_anndata``).

Every shared-contract test runs against both a dense (numpy) and a genuinely
sparse (scipy CSR) expression matrix via the ``sample`` fixture, so the two
branches in ``from_anndata`` are exercised symmetrically:

  * dense numpy is passed straight through (no copy, no sparsification);
  * scipy-sparse is converted to a pydata ``sparse.COO`` duck array so xarray
    can wrap it.

Path-specific guarantees that only make sense for one backing (COO structure
preservation, zero-copy passthrough) live in dedicated tests that skip the
irrelevant parametrization rather than asserting something vacuous.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import sparse

ad = pytest.importorskip("anndata")
import pandas as pd

from manykinds import LabeledArray
from manylatents.singlecell.data.adapters.formats.adapters import from_anndata


# Non-square on purpose: with N_CELLS != N_GENES a transposed or mis-mapped
# axis fails loudly (coord-length mismatch) instead of silently passing.
N_CELLS, N_GENES = 60, 40


def _make_adata(X):
    """Wrap an expression matrix in an AnnData with simple cell/gene names."""
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]),
    )


def _coords(adata):
    return {"cell": list(adata.obs_names), "gene": list(adata.var_names)}


def _dense(data):
    """Materialize a DataArray backing to a numpy array, sparse or not.

    ``from_anndata`` keeps dense input dense (numpy) and converts scipy-sparse
    input to a pydata ``sparse.COO``; this normalizes both for value comparison.
    """
    return np.asarray(data.todense()) if hasattr(data, "todense") else np.asarray(data)


def _sparse_dense_pair(seed=0):
    """Return ``(dense_ref, csr)`` where ~70% of the entries are zero.

    Reused by the fixture and by tests that need a fresh sparse matrix (layers).
    """
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((N_CELLS, N_GENES)).astype(np.float32)
    dense = (dense * (rng.random(dense.shape) < 0.3)).astype(np.float32)
    return dense, sp.csr_matrix(dense)


@pytest.fixture(params=["dense", "sparse"])
def sample(request):
    """``(adata, dense_ref)`` for a dense or scipy-sparse expression matrix.

    ``adata.X`` is a numpy array (``dense`` param) or a ``scipy.sparse`` CSR
    matrix (``sparse`` param); ``dense_ref`` is the numpy matrix the adapter's
    output must reconstruct to exactly.
    """
    if request.param == "sparse":
        dense, X = _sparse_dense_pair()
    else:
        dense = (
            np.random.default_rng(0)
            .standard_normal((N_CELLS, N_GENES))
            .astype(np.float32)
        )
        X = dense
    return _make_adata(X), dense


def _is_sparse_sample(adata):
    return sp.issparse(adata.X)


# --- basic conversion -------------------------------------------------------


def test_returns_labeled_array(sample):
    adata, _ = sample
    assert isinstance(from_anndata(adata, _coords(adata)), LabeledArray)


def test_dims_and_shape(sample):
    adata, _ = sample
    kind = from_anndata(adata, _coords(adata))
    assert kind.da.dims == ("cell", "gene")
    assert kind.da.shape == (N_CELLS, N_GENES)


def test_coords_preserved(sample):
    adata, _ = sample
    kind = from_anndata(adata, _coords(adata))
    assert list(kind.da.coords["cell"].values) == [f"c{i}" for i in range(N_CELLS)]
    assert list(kind.da.coords["gene"].values) == [f"g{i}" for i in range(N_GENES)]


def test_values_preserved(sample):
    adata, dense = sample
    kind = from_anndata(adata, _coords(adata))
    # Pure copy path (no arithmetic): values must be bit-exact, not merely close.
    np.testing.assert_array_equal(_dense(kind.da.data), dense)


def test_dtype_preserved(sample):
    """float32 input yields a float32 backing on both paths (no upcast to f64)."""
    adata, _ = sample
    kind = from_anndata(adata, _coords(adata))
    assert kind.da.dtype == np.float32
    assert kind.da.data.dtype == np.float32


def test_provenance_starts_empty(sample):
    adata, _ = sample
    assert from_anndata(adata, _coords(adata)).provenance == ()


# --- backing store: the dense/sparse split ---------------------------------


def test_backing_matches_input(sample):
    """Dense input stays numpy; scipy-sparse input becomes a ``sparse.COO``."""
    adata, _ = sample
    data = from_anndata(adata, _coords(adata)).da.data
    if _is_sparse_sample(adata):
        assert isinstance(data, sparse.COO)
    else:
        assert isinstance(data, np.ndarray) and not isinstance(data, sparse.COO)


def test_dense_backing_is_zero_copy(sample):
    """Dense input is threaded through untouched — no densify, no copy."""
    adata, _ = sample
    if _is_sparse_sample(adata):
        pytest.skip("passthrough guarantee is dense-specific")
    data = from_anndata(adata, _coords(adata)).da.data
    assert isinstance(data, np.ndarray)
    # from_anndata assigns ``data = X`` verbatim, so the buffer is shared.
    assert np.shares_memory(data, adata.X)


def test_sparse_backing_preserves_structure(sample):
    """CSR -> COO keeps nnz, fill value, dtype and every stored/implicit zero."""
    adata, dense = sample
    if not _is_sparse_sample(adata):
        pytest.skip("structure guarantees are sparse-specific")
    coo = from_anndata(adata, _coords(adata)).da.data
    assert isinstance(coo, sparse.COO)
    assert coo.nnz == adata.X.nnz          # no spurious fill-in
    assert coo.nnz < coo.size              # the fixture is genuinely sparse
    assert coo.fill_value == 0
    np.testing.assert_array_equal(coo.todense(), dense)  # exact, incl. zeros


def test_does_not_mutate_input(sample):
    """The adapter reads ``adata``; it must not add layers or alter X."""
    adata, _ = sample
    before = _dense(adata.X).copy()
    layers_before = set(adata.layers.keys())
    from_anndata(adata, _coords(adata))
    np.testing.assert_array_equal(_dense(adata.X), before)
    assert set(adata.layers.keys()) == layers_before


# --- empty rejection --------------------------------------------------------


@pytest.mark.parametrize(
    "make_X",
    [
        pytest.param(lambda: np.empty((N_CELLS, 0), dtype=np.float32), id="dense"),
        pytest.param(
            lambda: sp.csr_matrix((N_CELLS, 0), dtype=np.float32), id="sparse"
        ),
    ],
)
def test_empty_matrix_rejected(make_X):
    """A zero-size matrix is refused by ``LabeledArray.validate`` on both paths."""
    adata = _make_adata(make_X())
    with pytest.raises(ValueError, match="empty"):
        from_anndata(adata, {"cell": list(adata.obs_names), "gene": []})


# --- metadata ---------------------------------------------------------------


def test_metadata_becomes_attrs(sample):
    adata, _ = sample
    kind = from_anndata(adata, _coords(adata), metadata={"genome": "GRCh38"})
    assert kind.da.attrs["genome"] == "GRCh38"


def test_no_metadata_yields_empty_attrs(sample):
    adata, _ = sample
    assert from_anndata(adata, _coords(adata)).da.attrs == {}


# --- matrix selection precedence -------------------------------------------


def test_layer_selected_when_present(sample):
    adata, dense = sample
    adata.layers["counts"] = dense * 2
    kind = from_anndata(adata, _coords(adata), layer="counts")
    np.testing.assert_allclose(_dense(kind.da.data), dense * 2)


def test_sparse_layer_converts_to_coo(sample):
    """A scipy-sparse *layer* also takes the COO conversion branch."""
    adata, _ = sample
    layer_dense, layer_csr = _sparse_dense_pair(seed=7)
    adata.layers["sparse_counts"] = layer_csr
    kind = from_anndata(adata, _coords(adata), layer="sparse_counts")
    assert isinstance(kind.da.data, sparse.COO)
    np.testing.assert_array_equal(_dense(kind.da.data), layer_dense)


def test_missing_layer_falls_back_to_X(sample):
    adata, dense = sample
    kind = from_anndata(adata, _coords(adata), layer="nonexistent")
    np.testing.assert_array_equal(_dense(kind.da.data), dense)


def test_use_raw_selects_raw(sample):
    adata, dense = sample
    raw_X = dense * 3
    adata.raw = _make_adata(raw_X)
    kind = from_anndata(adata, _coords(adata), use_raw=True)
    np.testing.assert_allclose(_dense(kind.da.data), raw_X)


def test_use_raw_falls_back_to_X_when_no_raw(sample):
    adata, dense = sample
    assert adata.raw is None
    kind = from_anndata(adata, _coords(adata), use_raw=True)
    np.testing.assert_array_equal(_dense(kind.da.data), dense)


def test_raw_takes_precedence_over_layer(sample):
    """``use_raw`` wins over a provided layer (raw is checked first)."""
    adata, dense = sample
    adata.layers["counts"] = dense * 2
    adata.raw = _make_adata(dense * 3)
    kind = from_anndata(adata, _coords(adata), use_raw=True, layer="counts")
    np.testing.assert_allclose(_dense(kind.da.data), dense * 3)
