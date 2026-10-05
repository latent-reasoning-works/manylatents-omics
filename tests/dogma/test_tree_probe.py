"""Nested tree tuning uses chromosomes and deterministic model selection."""
import numpy as np
import pytest
from manylatents.dogma.benchmarks.probes import leave_chromosome_out_tree_probe
from manylatents.utils.exceptions import MeasurementUnavailable


def toy():
    rng = np.random.default_rng(9)
    chroms = np.repeat(['1', '2', '3', '4'], 50)
    labels = np.tile([0, 1], 100)
    x = rng.normal(size=(200, 3))
    x[:, 0] += labels * 2
    return x, labels, chroms


def run(x, y, c, **kwargs):
    return leave_chromosome_out_tree_probe(x, y, c, grid=((4, 5), (2, 5)),
                                           inner_folds=2, random_state=10, **kwargs)


def test_parallel_matches_serial():
    x, y, c = toy()
    a, b = run(x, y, c), run(x, y, c, n_jobs=2)
    np.testing.assert_array_equal(a.pop('scores'), b.pop('scores'))
    assert a == b
    assert set(a['chosen_config']) == set(c)
    assert all(len(t) == 2 for t in a['tune_auprc'].values())


def test_held_out_labels_do_not_enter_training_or_tuning():
    x, y, c = toy()
    a = run(x, y, c)
    y[c == '1'] = 1 - y[c == '1']
    b = run(x, y, c)
    np.testing.assert_array_equal(a['scores'][c == '1'], b['scores'][c == '1'])
    assert a['tune_auprc']['1'] == b['tune_auprc']['1']


def test_tie_prefers_smaller_model(monkeypatch):
    monkeypatch.setattr('sklearn.metrics.average_precision_score', lambda *args: .5)
    result = run(*toy())
    assert all(v == {'max_leaf_nodes': 2, 'max_iter': 5} for v in result['chosen_config'].values())


@pytest.mark.parametrize('kwargs', [{'grid': ()}, {'grid': ((1, 5),)}, {'inner_folds': 1},
                                    {'n_jobs': 0}, {'random_state': None}])
def test_invalid_parameters(kwargs):
    params = dict(random_state=0, grid=((2, 2),))
    params.update(kwargs)
    with pytest.raises(ValueError):
        leave_chromosome_out_tree_probe(*toy(), **params)


def test_undefined_data_and_shapes():
    x, y, c = toy()
    with pytest.raises(ValueError):
        run(x, y[:-1], c)
    with pytest.raises(MeasurementUnavailable):
        run(x, np.zeros(len(y)), c)
    x[0, 0] = np.nan
    with pytest.raises(MeasurementUnavailable):
        run(x, y, c)
