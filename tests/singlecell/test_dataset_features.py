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

