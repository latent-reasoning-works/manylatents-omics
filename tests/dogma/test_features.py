"""Reductions of the track matrix that published methods use."""
import numpy as np
import pytest

from manylatents.dogma.benchmarks.features import AVI_MODALITIES, category_pca, modality_max
from manylatents.utils.exceptions import MeasurementUnavailable


def test_modality_max_is_largest_absolute_score_per_scorer():
    values = np.array([[1.0, -3.0, 0.5], [0.0, 2.0, -4.0]])
    features, names = modality_max(values, np.array(["DNASE", "DNASE", "ATAC"]))
    assert names == ["DNASE", "ATAC"]
    np.testing.assert_array_equal(features, [[3.0, 0.5], [2.0, 4.0]])


def test_splicing_sums_its_three_maxima_and_needs_all_three():
    scorers = np.array(["SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS", "CAGE"])
    features, names = modality_max(np.array([[1.0, -2.0, 4.0, 9.0]]), scorers)
    assert names == ["CAGE", "SPLICING"]
    np.testing.assert_array_equal(features, [[9.0, 7.0]])
    _, names = modality_max(np.array([[1.0, -2.0, 9.0]]), scorers[[0, 1, 3]])
    assert names == ["CAGE"]
    assert len(AVI_MODALITIES) == 10


def test_modality_max_refuses_mismatched_or_unknown_tracks():
    with pytest.raises(ValueError, match="one scorer name per track"):
        modality_max(np.zeros((2, 3)), np.array(["DNASE", "DNASE"]))
    with pytest.raises(ValueError, match="no modality"):
        modality_max(np.zeros((2, 1)), np.array(["OTHER"]))


def test_category_pca_fits_on_the_reference_only():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(50, 6))
    query = rng.normal(size=(4, 6))
    scorers = np.array(["A"] * 4 + ["B"] * 2)
    q, r, names = category_pca(query, reference, scorers, n_components=3)
    assert names == ["A_pc1", "A_pc2", "A_pc3", "B_pc1", "B_pc2"]
    assert q.shape == (4, 5) and r.shape == (50, 5)
    np.testing.assert_allclose(r.mean(axis=0), 0, atol=1e-9)
    assert abs(np.corrcoef(r[:, 0], r[:, 1])[0, 1]) < 1e-8
    q2, _, _ = category_pca(query * 1.0, reference, scorers, n_components=3)
    np.testing.assert_array_equal(q, q2)
    other, _, _ = category_pca(np.vstack([query, query[:1] * 100]), reference, scorers, n_components=3)
    np.testing.assert_allclose(other[:4], q)


def test_category_pca_skips_a_category_the_reference_does_not_vary_in():
    rng = np.random.default_rng(1)
    reference = np.column_stack([rng.normal(size=(20, 2)), np.ones((20, 2))])
    query = rng.normal(size=(3, 4))
    q, r, names = category_pca(query, reference, np.array(["A", "A", "B", "B"]), n_components=2)
    assert names == ["A_pc1", "A_pc2"]
    assert q.shape == (3, 2)


def test_omitted_tracks_are_counted_without_changing_tuple_unpacking():
    result = modality_max([[1, 2, 3]], ["CAGE", "SPLICE_SITES", "OTHER"])
    assert result.n_dropped == {"CAGE": 0, "SPLICE_SITES": 1, "OTHER": 1}
    assert len(result) == 2
    reference = np.column_stack([np.arange(20), np.full(20, 0.1), np.ones(20)])
    result = category_pca([[2, 99, 100]], reference, ["A", "A", "B"])
    assert result.n_dropped == {"A": 1, "B": 1}
    assert result[2] == ["A_pc1"]
    np.testing.assert_allclose(result[0], category_pca([[2, -99, -100]], reference, ["A", "A", "B"])[0])


@pytest.mark.parametrize("reference", [np.ones((10, 2)), np.ones((1, 2)), np.empty((0, 2))])
def test_undefined_pca_raises_measurement_unavailable(reference):
    with pytest.raises(MeasurementUnavailable):
        category_pca([[1, 2]], reference, ["A", "B"])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_features_are_not_hidden_by_reductions(bad):
    with pytest.raises(MeasurementUnavailable, match="finite"):
        modality_max([[1, bad]], ["CAGE", "OTHER"])
    with pytest.raises(MeasurementUnavailable, match="finite"):
        category_pca([[1, bad]], [[1, 0], [2, 0]], ["A", "B"])
    with pytest.raises(MeasurementUnavailable, match="finite"):
        category_pca([[1, 0]], [[1, 0], [2, bad]], ["A", "B"])


@pytest.mark.parametrize("n_components", [0, -1, 1.5, np.nan])
def test_pca_requires_a_positive_integer_component_count(n_components):
    with pytest.raises(ValueError, match="positive integer"):
        category_pca([[1]], [[1], [2]], ["A"], n_components=n_components)


def test_splicing_takes_maxima_before_summing():
    result, names = modality_max([[-3, 2, -5, 4, -7, 6]],
                                ["SPLICE_SITES"] * 2 + ["SPLICE_SITE_USAGE"] * 2
                                + ["SPLICE_JUNCTIONS"] * 2)
    assert names == ["SPLICING"]
    np.testing.assert_array_equal(result, [[15]])


def test_nonfinite_pca_projection_is_unavailable():
    query = np.full((1, 4), 1e308)
    reference = np.array([[-1.] * 4, [1.] * 4])
    with pytest.raises(MeasurementUnavailable, match="finite"):
        category_pca(query, reference, ["A"] * 4)
