"""Genomic block labels and vectorised site proximity."""
import numpy as np
import pandas as pd
import pytest

from manylatents.dogma.benchmarks.loci import genomic_blocks, nearest_site_distance
from manylatents.utils.exceptions import MeasurementUnavailable


def test_genomic_blocks_normalize_prefix_and_respect_boundaries():
    variants = pd.DataFrame({'chrom': ['1', 'chr1', '2', 'X'],
                             'pos': [99_999, 100_000, 1_234_567, 0]})
    assert genomic_blocks(variants).tolist() == ['chr1:0', 'chr1:1', 'chr2:12', 'chrX:0']
    assert genomic_blocks(variants, size=10)[0] == 'chr1:9999'
    assert genomic_blocks(variants.iloc[:0]).shape == (0,)


@pytest.mark.parametrize('size', [0, -1, 1.5, True])
def test_invalid_block_size(size):
    with pytest.raises(ValueError, match='positive int'):
        genomic_blocks(pd.DataFrame({'chrom': ['1'], 'pos': [10]}), size)


def test_nearest_sites_unsorted_ties_exact_and_endpoints():
    actual = nearest_site_distance(
        ['chr1', '1', '1', '1', 'chr2', '2'], [20, 5, 40, 30, 70, 100],
        ['1', 'chr2', 'chr1', '2', '1'], [30, 90, 10, 50, 30])
    np.testing.assert_array_equal(actual, [10, 5, 10, 0, 20, 10])
    assert actual.dtype == np.int64


def test_missing_sites_raise_with_all_missing_chromosomes():
    with pytest.raises(MeasurementUnavailable, match='2.*X'):
        nearest_site_distance(['1', 'chr2', 'X'], [1, 2, 3], ['chr1'], [1])
    with pytest.raises(MeasurementUnavailable, match='1'):
        nearest_site_distance(['1'], [1], [], [])


def test_empty_queries_and_shape_validation():
    assert nearest_site_distance([], [], ['1'], [1]).dtype == np.int64
    for args in [(['1'], [], ['1'], [1]), (['1'], [1], ['1'], []),
                 (['1'], [1.5], ['1'], [1])]:
        with pytest.raises(ValueError):
            nearest_site_distance(*args)


def test_searchsorted_matches_brute_force():
    rng = np.random.default_rng(40)
    queries, sites = rng.integers(0, 10000, 100), rng.integers(0, 10000, 300)
    actual = nearest_site_distance(['1'] * len(queries), queries, ['chr1'] * len(sites), sites)
    np.testing.assert_array_equal(actual, np.abs(queries[:, None] - sites).min(axis=1))
