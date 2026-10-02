"""Paired cluster bootstrap for pooled average precision differences."""
import numpy as np
import pytest

pytest.importorskip('sklearn')
from manylatents.dogma.benchmarks.intervals import clustered_average_precision_difference
from manylatents.utils.exceptions import MeasurementUnavailable


def compare(a, b, y, clusters, seed=1, n_bootstrap=300):
    return clustered_average_precision_difference(
        a, b, y, clusters, n_bootstrap=n_bootstrap, rng=np.random.default_rng(seed))


def data():
    rng = np.random.default_rng(30)
    y = np.tile([False, True], 100)
    return y + rng.normal(0, 0.2, 200), rng.normal(size=200), y, np.repeat(np.arange(20), 10)


def test_identical_scores_have_zero_difference_and_interval():
    a, _, y, clusters = data()
    result = compare(a, a, y, clusters)
    assert result['difference'] == result['se'] == 0
    np.testing.assert_array_equal(result['ci95'], [0, 0])
    assert result['n_clusters_positive'] == result['n_clusters_negative'] == 20


def test_better_score_has_positive_interval_and_is_deterministic():
    args = data()
    result = compare(*args)
    assert result['ci95'][0] > 0
    assert result == compare(*args)
    assert result['difference'] == result['auprc_a'] - result['auprc_b']


def test_clustered_positives_widen_interval_relative_to_row_bootstrap():
    rng = np.random.default_rng(10)
    y = np.r_[np.ones(80), np.zeros(240)]
    clusters = np.r_[np.repeat(np.arange(4), 20), np.arange(4, 244)]
    a = np.r_[np.repeat([3., 1., -1., -3.], 20), rng.normal(size=240)]
    b = np.r_[np.full(80, 0.5), rng.normal(size=240)]
    clustered = compare(a, b, y, clusters, n_bootstrap=500)
    rows = compare(a, b, y, np.arange(len(y)), n_bootstrap=500)
    assert np.ptp(clustered['ci95']) > np.ptp(rows['ci95']) * 2


def test_mixed_clusters_resample_class_subsets_with_the_same_paired_draw():
    from sklearn.metrics import average_precision_score

    a, b, y, clusters = data()
    rng = np.random.default_rng(12)
    differences = []
    for _ in range(30):
        rows = []
        for label in (True, False):
            ids = np.unique(clusters[y == label])
            for cluster in rng.choice(ids, size=len(ids), replace=True):
                rows.extend(np.flatnonzero((clusters == cluster) & (y == label)))
        differences.append(average_precision_score(y[rows], a[rows])
                           - average_precision_score(y[rows], b[rows]))
    result = compare(a, b, y, clusters, seed=12, n_bootstrap=30)
    np.testing.assert_allclose(result['ci95'], np.percentile(differences, [2.5, 97.5]))
    assert result['se'] == pytest.approx(np.std(differences, ddof=1))


@pytest.mark.parametrize('case', ['nonfinite_a', 'nonfinite_b', 'one_class', 'few_positive',
                                 'few_negative', 'invalid_label'])
def test_undefined_inputs(case):
    a, b, y, c = data()
    if case == 'nonfinite_a':
        a[0] = np.nan
    elif case == 'nonfinite_b':
        b[0] = np.inf
    elif case == 'one_class':
        y[:] = True
    elif case == 'few_positive':
        c[y] = 0
    elif case == 'few_negative':
        c[~y] = 0
    else:
        y = y.astype(float)
        y[0] = 2
    with pytest.raises(MeasurementUnavailable):
        compare(a, b, y, c)


def test_lengths_generator_and_bootstrap_count_are_validated():
    a, b, y, c = data()
    with pytest.raises(ValueError, match='length'):
        compare(a[:-1], b, y, c)
    with pytest.raises(TypeError, match='Generator'):
        clustered_average_precision_difference(a, b, y, c, rng=1)
    for n in [0, 1, 1.5, True]:
        with pytest.raises(ValueError, match='n_bootstrap'):
            compare(a, b, y, c, n_bootstrap=n)
