"""Contracts for read-only dataset descriptions."""

import numpy as np
import pytest
import scipy.sparse as sp

from manylatents.singlecell import dataset_features as df


def _counts(n=400, p=50, seed=1):
    return np.random.default_rng(seed).poisson(1.5, size=(n, p)).astype(np.float32)


def test_counts_like_raw_counts_is_raw_integer():
    assert df.counts_like(sp.csr_matrix(_counts())) == "raw_integer"


def test_counts_like_log1p_is_processed():
    assert df.counts_like(np.log1p(_counts())) == "processed"


def test_counts_like_signed_values_are_processed():
    matrix = _counts()
    matrix[0, 0] = -1.0
    assert df.counts_like(matrix) == "processed"


def test_counts_like_scaled_integer_data_is_raw_integer():
    assert df.counts_like(_counts() * 2) == "raw_integer"


def test_counts_like_size_factor_scaled_is_processed():
    matrix = _counts()
    matrix = matrix / (matrix.sum(1, keepdims=True) + 1) * 1e4
    assert df.counts_like(matrix) == "processed"


def test_counts_like_small_noninteger_fraction_is_unknown():
    matrix = _counts(n=2000, p=50)
    nonzero = np.flatnonzero(matrix)
    matrix.flat[nonzero[:len(nonzero) // 200]] += 0.5
    assert df.counts_like(matrix) == "unknown"


def test_counts_like_few_nonzero_values_is_undefined():
    matrix = np.zeros((100, 100))
    matrix.flat[:999] = 1.0
    assert df.counts_like(matrix) == "undefined"


def test_counts_like_ignores_stored_zeros():
    matrix = sp.csr_matrix(_counts())
    matrix.data[:10] = 0.0
    assert df.counts_like(matrix) == "raw_integer"


def test_counts_like_never_densifies_a_sparse_matrix(monkeypatch):
    matrix = sp.csr_matrix(_counts())
    monkeypatch.setattr(type(matrix), "toarray", lambda *a, **k: pytest.fail("densified"))
    assert df.counts_like(matrix) == "raw_integer"


def test_counts_like_is_deterministic_above_the_sample_cap():
    matrix = sp.random(2000, 200, density=0.5, format="csr", random_state=0,
                       data_rvs=lambda n: np.ones(n))
    assert df.counts_like(matrix) == df.counts_like(matrix.copy()) == "raw_integer"


@pytest.mark.parametrize("fraction,expected", [(0, "raw_integer"), (0.01, "unknown"),
                                               (0.011, "processed")])
def test_counts_like_fraction_boundary(fraction, expected):
    matrix = np.ones((100, 10))
    matrix.flat[:int(fraction * matrix.size)] += 0.5
    assert df.counts_like(matrix) == expected


def test_counts_like_constants_pin_the_definition():
    assert df.COUNTS_LIKE_VERSION == "counts_like/1"
    assert df.COUNTS_LIKE_MAX_VALUES == 100_000
    assert df.COUNTS_LIKE_MIN_VALUES == 1_000
    assert df.COUNTS_LIKE_SEED == 0
    assert df.COUNTS_LIKE_TOL == 1e-6
    assert df.COUNTS_LIKE_PROCESSED_FRACTION == 0.01


@pytest.mark.parametrize("constructor", [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix,
                                         sp.lil_matrix, sp.dok_matrix, sp.csr_array])
def test_counts_like_supports_sparse_formats(constructor):
    assert df.counts_like(constructor(_counts())) == "raw_integer"


def test_counts_like_uses_matrix_values_for_duplicate_entries():
    indices = np.repeat(np.arange(1000), 2)
    matrix = sp.coo_matrix((np.full(2000, 0.5), (indices, np.zeros(2000, dtype=int))),
                           shape=(1000, 1))
    before = matrix.data.copy()
    assert df.counts_like(matrix) == "raw_integer"
    np.testing.assert_array_equal(matrix.data, before)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_counts_like_nonfinite_is_unknown(value):
    matrix = np.ones((100, 10))
    matrix[0, 0] = value
    assert df.counts_like(matrix) == "unknown"


def _adata():
    from anndata import AnnData

    matrix = sp.csr_matrix(_counts(300, 40))
    matrix.data[:3] = 0
    data = AnnData(matrix)
    data.var_names = [f"MT-{i}" if i < 3 else f"GENE{i}" for i in range(40)]
    for layer in ("counts", "spliced", "unspliced"):
        data.layers[layer] = matrix.copy()
    return data


def _rows(data, roles=None, **kwargs):
    return {row["key"]: row for row in df.dataset_features(data, roles or {}, **kwargs)}


EXPECTED_KEYS = (
    ("n_cells", "int"), ("n_genes", "int"), ("tested_matrix", "str"),
    ("counts_like", "enum"), ("sparsity", "float"), ("modality", "enum"),
    ("technology", "enum"), ("species", "enum"), ("has_mito_genes", "bool"),
    ("technical_batch_column", "str"), ("n_technical_batches", "int"),
    ("sample_column", "str"), ("n_samples", "int"), ("has_batch_column", "bool"),
    ("has_condition_column", "bool"), ("n_condition_levels", "int"),
    ("samples_per_condition_min", "int"), ("has_replicates", "bool"),
    ("has_time_column", "bool"), ("n_timepoints", "int"),
    ("samples_per_timepoint_min", "int"), ("has_cell_labels", "bool"),
    ("batch_condition_confounded", "bool"), ("batch_time_confounded", "bool"),
    ("median_genes_per_cell", "float"), ("spatial_resolution", "enum"),
    ("has_spliced_unspliced_layers", "bool"),
)
ROLE_KEYS = [key for key, _ in EXPECTED_KEYS[9:24]]
GENE_KEYS = ["n_genes", "modality", "technology", "species", "has_mito_genes",
             "median_genes_per_cell", "spatial_resolution", "has_spliced_unspliced_layers"]


def test_rows_follow_the_row_schema():
    rows = df.dataset_features(_adata(), {})
    assert df.KEYS == EXPECTED_KEYS
    assert [row["key"] for row in rows] == [key for key, _ in EXPECTED_KEYS]
    assert len(rows) == 27
    for row, (_, kind) in zip(rows, EXPECTED_KEYS):
        assert set(row) == set(df.FIELDS) == {
            "key", "value", "type", "status", "reason", "source", "definition_version"
        }
        assert row["definition_version"] == "dataset_features/1"
        assert row["status"] in df.STATUSES
        assert row["type"] == kind
        if row["status"] == "ok":
            assert row["reason"] is None
        else:
            assert row["value"] is None
            assert isinstance(row["reason"], str) and row["reason"]


@pytest.mark.parametrize("array", [False, True])
def test_values_are_native_python(array):
    data = _counts() if array else _adata()
    for row in df.dataset_features(data, {}):
        assert type(row["value"]) in (bool, int, float, str, type(None))


def test_data_keys_on_anndata():
    data = _adata()
    rows = _rows(data)
    expected = {"n_cells": 300, "n_genes": 40, "tested_matrix": "X",
                "counts_like": "raw_integer", "has_mito_genes": True,
                "has_spliced_unspliced_layers": True,
                "sparsity": pytest.approx(1 - np.count_nonzero(data.X.data) / 12000),
                "median_genes_per_cell": float(np.median(np.asarray((data.X != 0).sum(1)).ravel()))}
    for key, value in expected.items():
        assert rows[key]["value"] == value
        assert rows[key]["status"] == "ok"
        assert rows[key]["source"] == "data"


def test_median_genes_counts_nonzero_not_positive():
    from anndata import AnnData

    rows = _rows(AnnData(sp.csr_matrix([[-1., 0, 1], [-2, -3, 4]])))
    assert rows["median_genes_per_cell"]["value"] == 2.5


def test_layer_parameter_reads_the_named_layer():
    data = _adata()
    data.X = data.X.log1p()
    rows = _rows(data, layer="counts")
    assert rows["tested_matrix"]["value"] == "layers:counts"
    assert rows["counts_like"]["value"] == "raw_integer"
    assert _rows(data)["counts_like"]["value"] == "processed"


@pytest.mark.parametrize("array", [False, True])
def test_unknown_layer_raises(array):
    with pytest.raises(ValueError, match="layer"):
        _rows(_counts() if array else _adata(), layer="missing")


@pytest.mark.parametrize("sparse_input", [False, True])
def test_point_cloud_gives_the_point_cloud_column(sparse_input):
    matrix = np.random.default_rng(0).normal(size=(200, 5))
    matrix -= matrix.mean(0)
    rows = _rows(sp.csr_matrix(matrix) if sparse_input else matrix)
    for key, value in {"n_cells": 200, "tested_matrix": "X", "counts_like": "processed",
                       "sparsity": 0.0}.items():
        assert rows[key]["value"] == value
        assert rows[key]["status"] == "ok"
    for key in GENE_KEYS:
        assert rows[key]["status"] == "not_applicable"
        assert rows[key]["value"] is None
    for key in ROLE_KEYS:
        assert rows[key]["status"] == "not_supplied"


def test_dataset_features_never_densifies(monkeypatch):
    data = _adata()
    monkeypatch.setattr(sp.csr_matrix, "toarray", lambda *a, **k: pytest.fail("densified"))
    for layer in (None, *data.layers):
        assert len(df.dataset_features(data, {}, layer=layer)) == 27


def test_input_is_not_mutated():
    from pandas.testing import assert_frame_equal

    data = _adata()
    data.obs["sample"] = "one"
    data.uns["nested"] = {"species": "human"}
    data.obsm["coordinates"] = np.ones((300, 2))
    before = data.copy()
    _rows(data, {"sample": "sample"}, layer="counts")
    for left, right in [(data.X, before.X), *[(data.layers[k], before.layers[k]) for k in data.layers]]:
        for attr in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(left, attr), getattr(right, attr))
    assert_frame_equal(data.obs, before.obs)
    assert_frame_equal(data.var, before.var)
    assert data.uns == before.uns
    np.testing.assert_array_equal(data.obsm["coordinates"], before.obsm["coordinates"])


@pytest.mark.parametrize("data,roles,error", [
    (np.ones((2, 2)), {"species": "human"}, ValueError),
    (np.ones((2, 2)), {"sample": 3}, ValueError),
    (np.ones(2), {}, TypeError), (np.ones((2, 2, 2)), {}, TypeError),
    ([[1, 2]], {}, TypeError), (object(), {}, TypeError),
])
def test_bad_inputs_raise(data, roles, error):
    with pytest.raises(error):
        df.dataset_features(data, roles)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
def test_empty_matrix_has_undefined_sparsity(shape):
    rows = _rows(np.zeros(shape))
    assert rows["sparsity"]["status"] == "undefined"
    assert rows["counts_like"]["status"] == "undefined"


def test_small_matrix_has_undefined_counts_with_a_reason():
    row = _rows(np.ones((10, 10)))["counts_like"]
    assert row["status"] == "undefined"
    assert row["value"] is None
    assert row["reason"]
