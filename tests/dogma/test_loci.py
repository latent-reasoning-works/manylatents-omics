"""Small, explicit genomic neighbourhoods."""
import numpy as np
import pandas as pd
import pytest

from manylatents.dogma.benchmarks.loci import (
    chain_loci, distant_neighbour_distances, genomic_separation, positional_scores,
)


@pytest.fixture
def variants():
    return pd.DataFrame({
        "chrom": ["1", "1", "1", "1", "1", "2"],
        "pos": [100, 105, 1_000_105, 200, 3_000_000, 100],
        "label": [True, True, True, False, False, False],
        "match_group": ["a", "b", "c", "a", "c", "b"],
    })


def test_chain_and_separation(variants):
    np.testing.assert_array_equal(chain_loci(variants), [0, 0, 1, 0, 1, 0])
    separation = genomic_separation(variants)
    assert separation.dtype == float
    assert separation[0, 1] == 5
    assert np.isinf(separation[-1]).all()
    assert np.isinf(separation.diagonal()).all()
    for column in variants:
        if column in ("chrom", "pos", "label", "match_group"):
            with pytest.raises(ValueError, match=column):
                chain_loci(variants.drop(columns=column))
    variants.loc[3, "match_group"] = "missing"
    with pytest.raises(ValueError, match="no positive"):
        chain_loci(variants)


@pytest.mark.parametrize("exclusion", [0, 1000, np.inf])
def test_neighbours_against_brute_force(variants, exclusion):
    distance = np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]).astype(float)
    np.fill_diagonal(distance, np.inf)
    original = distance.copy()
    expected = []
    for i, row in variants.iterrows():
        candidates = []
        for j, other in variants.iterrows():
            allowed = (exclusion == 0 or row.chrom != other.chrom or
                       abs(row.pos - other.pos) > exclusion)
            candidates.append(distance[i, j] if allowed else np.inf)
        expected.append(sorted(candidates)[:3])
    actual = distant_neighbour_distances(distance, genomic_separation(variants), exclusion, 3)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(distance, original)


def test_positional_scores(variants):
    scores = positional_scores(variants)
    assert set(scores) == {"n_cohort_within_1kb", "neg_log_nearest_cohort",
                           "neg_log_nearest_positive_uses_labels"}
    assert all(score[0] > score[4] for score in scores.values())
