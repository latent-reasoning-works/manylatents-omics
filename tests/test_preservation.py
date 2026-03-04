"""Tests for popgen preservation metrics.

Regression tests ensure that GeographicPreservation and AdmixturePreservation
accept the ``cache`` kwarg passed by manylatents core (v0.1.2+) without error.
"""

import numpy as np
import pandas as pd
import pytest

from manylatents.popgen.metrics.preservation import (
    AdmixturePreservation,
    GeographicPreservation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 50  # small enough for fast geodesic computation
RNG = np.random.default_rng(0)


class _FakeDataset:
    """Minimal dataset stub with attributes required by preservation metrics."""

    def __init__(self):
        # Lat/lon in radians-friendly range
        self.latitude = pd.Series(RNG.uniform(-60, 60, size=N), name="latitude")
        self.longitude = pd.Series(RNG.uniform(-180, 180, size=N), name="longitude")
        self.geographic_preservation_indices = None  # use all samples

        self.population_label = pd.Series(
            [f"pop_{i % 5}" for i in range(N)], name="Population"
        )

        # Two admixture K values (K=3 and K=5)
        self.admixture_ratios = {
            3: self._make_admixture(3),
            5: self._make_admixture(5),
        }

    @staticmethod
    def _make_admixture(k: int) -> pd.DataFrame:
        props = RNG.dirichlet(np.ones(k), size=N)
        df = pd.DataFrame(props, columns=[f"component_{i}" for i in range(k)])
        return df


@pytest.fixture
def dataset():
    return _FakeDataset()


@pytest.fixture
def embeddings():
    return RNG.standard_normal((N, 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# Regression: cache kwarg must not raise TypeError
# ---------------------------------------------------------------------------

def test_geographic_preservation_accepts_cache(embeddings, dataset):
    """manylatents core passes cache= to all metrics; must not raise TypeError."""
    result = GeographicPreservation(embeddings, dataset, cache={})
    assert result is None or isinstance(result, float)


def test_admixture_preservation_accepts_cache_single_k(embeddings, dataset):
    """AdmixturePreservation with admixture_k accepts cache= without error."""
    result = AdmixturePreservation(embeddings, dataset, admixture_k=3, cache={})
    assert result is None or isinstance(result, float)


def test_admixture_preservation_accepts_cache_all_k(embeddings, dataset):
    """AdmixturePreservation over all Ks accepts cache= without error."""
    result = AdmixturePreservation(embeddings, dataset, cache={})
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(dataset.admixture_ratios),)


# ---------------------------------------------------------------------------
# Basic sanity checks
# ---------------------------------------------------------------------------

def test_geographic_preservation_returns_float(embeddings, dataset):
    result = GeographicPreservation(embeddings, dataset)
    assert result is None or isinstance(result, float)


def test_admixture_preservation_single_k_returns_float(embeddings, dataset):
    result = AdmixturePreservation(embeddings, dataset, admixture_k=3)
    assert result is None or isinstance(result, float)


def test_admixture_preservation_all_k_shape(embeddings, dataset):
    result = AdmixturePreservation(embeddings, dataset)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(dataset.admixture_ratios),)


def test_admixture_preservation_invalid_k_raises(embeddings, dataset):
    with pytest.raises(ValueError, match="not found"):
        AdmixturePreservation(embeddings, dataset, admixture_k=99)


def test_admixture_preservation_subsampling(embeddings, dataset):
    """max_samples subsampling should still return a valid result."""
    result = AdmixturePreservation(
        embeddings, dataset, admixture_k=3, max_samples=20, random_seed=0
    )
    assert result is None or isinstance(result, float)
