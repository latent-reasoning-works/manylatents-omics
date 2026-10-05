"""Reference-only reductions and explicit exclusions."""
import numpy as np
import pytest
from manylatents.dogma.benchmarks.features import (
    signed_extremes, modality_breadth, normalised_maxima, group_maxima,
)
from manylatents.utils.exceptions import MeasurementUnavailable


def test_signed_extremes_preserve_direction_and_count_unsigned_tracks():
    result = signed_extremes([[2, -4, 9], [-3, -1, 8], [1, 2, 7]], ['A', 'A', 'B'], {'A'})
    assert result[1] == ['A_positive', 'A_negative']
    np.testing.assert_array_equal(result[0], [[2, -4], [0, -3], [2, 0]])
    assert result.n_dropped == {'A': 0, 'B': 1}


def test_reference_reductions_drop_constants_and_use_reference_moments():
    reference = np.array([[-1, -2, .1, 7], [1, 2, .1, 7]])
    query = [[0, 0, 99, 8], [2, 2, 99, 8]]
    scorers = ['A', 'A', 'A', 'B']
    breadth = modality_breadth(query, reference, scorers)
    assert breadth[1] == ['A_fraction_above_background', 'A_effective_fraction']
    np.testing.assert_allclose(breadth[0], [[0, 0], [.5, 25/34]])
    maxima = normalised_maxima(query, reference, scorers)
    np.testing.assert_allclose(maxima[0], [[0], [2]])
    assert maxima[1] == ['A_normalised_max']
    assert maxima.n_dropped == breadth.n_dropped == {'A': 1, 'B': 1}
    more = normalised_maxima(query + [[1e6]*4], reference, scorers)
    np.testing.assert_array_equal(more[0][:2], maxima[0])


def test_group_maxima_only_returns_present_combinations():
    result = group_maxima([[1, -4, 2, 3]], ['A', 'A', 'A', 'B'], ['x', 'x', 'y', 'y'])
    assert result[1] == ['A::x', 'A::y', 'B::y']
    np.testing.assert_array_equal(result[0], [[4, 2, 3]])
    assert result.n_dropped == {'A': 0, 'B': 0}


@pytest.mark.parametrize('function', [modality_breadth, normalised_maxima])
def test_reference_errors_are_explicit(function):
    with pytest.raises(ValueError):
        function([[1, 2]], [[1], [2]], ['A', 'A'])
    for reference in ([], [[1, 1]], [[1, 1], [1, 1]], [[1, np.nan], [2, 3]]):
        with pytest.raises((ValueError, MeasurementUnavailable)):
            function([[1, 2]], reference, ['A', 'A'])


@pytest.mark.parametrize('quantile', [-.1, 1.1, np.nan])
def test_invalid_quantile(quantile):
    with pytest.raises(ValueError):
        modality_breadth([[1]], [[0], [1]], ['A'], quantile=quantile)


def test_shapes_nonfinite_and_missing_groups():
    with pytest.raises(ValueError):
        group_maxima([[1, 2]], ['A', 'A'], ['x'])
    with pytest.raises(MeasurementUnavailable):
        group_maxima([[1, 2]], ['A', 'A'], ['x', None])
    with pytest.raises(MeasurementUnavailable):
        signed_extremes([[np.nan]], ['A'], {'A'})
    with pytest.raises(MeasurementUnavailable):
        signed_extremes([[1]], ['A'], {'B'})


def test_effective_fraction_avoids_overflow_for_finite_standardised_values():
    result = modality_breadth([[1e200, 1e200]], [[-1, -1], [1, 1]], ['A', 'A'])
    np.testing.assert_allclose(result[0], [[1, 1]])
