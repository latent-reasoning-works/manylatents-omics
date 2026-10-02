"""Nested grouped probes: held-out rows never inform their own model."""
import numpy as np
import pytest

pytest.importorskip('sklearn')
from sklearn.metrics import average_precision_score
from manylatents.dogma.benchmarks.probes import leave_chromosome_out_probe
from manylatents.utils.exceptions import MeasurementUnavailable


def toy():
    rng = np.random.default_rng(23)
    chroms = np.repeat(['1', '2', '3', '4', '5'], 40)
    labels = np.tile([False, True], 100)
    features = rng.normal(size=(200, 3))
    features[:, 0] += labels * 4
    return features, labels, chroms


def probe(features, labels, chroms, **kwargs):
    return leave_chromosome_out_probe(features, labels, chroms, c_grid=[0.01, 0.1], **kwargs)


def test_recovers_signal_and_normalizes_chromosomes():
    x, y, c = toy()
    result = probe(x, y.astype(object), [' CHR' + name for name in c])
    assert result['scores'].shape == (200,)
    assert average_precision_score(y, result['scores']) > 0.9
    assert result['n_chromosomes'] == 5
    assert set(result['chosen_c']) == set(c)
    assert all(value == 3 for value in result['n_features_used'].values())
    assert all(result['chosen_c_on_edge'].values())


def test_held_out_labels_cannot_change_held_out_scores():
    x, y, c = toy()
    base = probe(x, y, c)
    y[c == '1'] = ~y[c == '1']
    changed = probe(x, y, c)
    np.testing.assert_array_equal(base['scores'][c == '1'], changed['scores'][c == '1'])
    assert base['chosen_c']['1'] == changed['chosen_c']['1']


def test_held_out_outlier_changes_only_its_own_prediction_in_that_fold():
    x, y, c = toy()
    base = probe(x, y, c)
    x[0] *= 1000
    changed = probe(x, y, c)
    np.testing.assert_array_equal(base['scores'][1:40], changed['scores'][1:40])
    assert base['scores'][0] != changed['scores'][0]
    assert base['chosen_c']['1'] == changed['chosen_c']['1']


def test_training_constant_is_dropped_and_single_class_holdout_is_scored():
    x, y, c = toy()
    x[c != '1', 1] = 0.1
    y[c == '1'] = False
    result = probe(x, y, c)
    assert result['n_features_used']['1'] == 2
    assert result['n_features_dropped']['1'] == 1
    assert np.isfinite(result['scores']).all()


def test_parallel_is_bitwise_identical():
    x, y, c = toy()
    serial = probe(x, y, c, n_jobs=1)
    parallel = probe(x, y, c, n_jobs=2)
    for key in serial:
        if key == 'scores':
            np.testing.assert_array_equal(serial[key], parallel[key])
        else:
            assert serial[key] == parallel[key]


@pytest.mark.parametrize('case', ['one_class', 'no_inner', 'constant', 'nonfinite', 'invalid_label', 'empty'])
def test_undefined_measurements_raise(case):
    x, y, c = toy()
    if case == 'one_class':
        y[c != '1'] = False
    elif case == 'no_inner':
        x, y, c = np.array([[0.], [1.], [2.]]), [0, 1, 0], ['1', '2', '3']
    elif case == 'constant':
        x[:] = 0
    elif case == 'nonfinite':
        x[0, 0] = np.nan
    elif case == 'invalid_label':
        y = y.astype(float)
        y[0] = 2
    else:
        x, y, c = x[:0], y[:0], c[:0]
    with pytest.raises(MeasurementUnavailable):
        probe(x, y, c)


def test_invalid_folds_are_skipped_and_counted():
    from manylatents.dogma.benchmarks.probes import _leave_chromosome_out

    rng = np.random.default_rng(4)
    c = np.repeat(['1', '2', '3', '4'], 20)
    y = np.isin(c, ['1', '2'])
    x = rng.normal(size=(80, 2)) + np.asarray(y)[:, None]
    # Holding out 4 leaves a single negative chromosome. Its inner fold is
    # skipped; the remaining positive validation rows have well-defined AP=1.
    _, details = _leave_chromosome_out('4', x, y, c, [0.01, 0.1], 10, 2000)
    assert details['n_inner_folds_skipped'] == 1
    assert details['n_inner_rows_skipped'] == 20
    assert details['chosen_c'] == 0.01
    # The symmetric fold leaves only negative validation rows: AP is undefined.
    with pytest.raises(MeasurementUnavailable, match='no positives'):
        _leave_chromosome_out('1', x, y, c, [0.01, 0.1], 10, 2000)


@pytest.mark.parametrize('kwargs', [{'c_grid': []}, {'inner_folds': 1}, {'inner_folds': 2.5}, {'n_jobs': 0}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        leave_chromosome_out_probe(*toy(), **kwargs)


def test_mismatched_lengths():
    x, y, c = toy()
    with pytest.raises(ValueError, match='same length'):
        probe(x, y[:-1], c)


def test_nested_tuning_matches_independent_grouped_reference():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    x, y, c = toy()
    x[c == '2'] *= 10
    result = probe(x, y, c, inner_folds=3)
    train = c != '1'
    tx, ty, groups = x[train], y[train], c[train]
    aps = {}
    for strength in [0.01, 0.1]:
        decisions = np.empty(len(tx))
        for fit, valid in GroupKFold(3).split(tx, ty, groups):
            mean, std = tx[fit].mean(axis=0), tx[fit].std(axis=0)
            model = LogisticRegression(C=strength, class_weight='balanced', max_iter=2000)
            model.fit((tx[fit] - mean) / std, ty[fit])
            decisions[valid] = model.decision_function((tx[valid] - mean) / std)
        aps[strength] = average_precision_score(ty, decisions)
    chosen = max(aps, key=aps.get)
    assert result['chosen_c']['1'] == chosen
    mean, std = tx.mean(axis=0), tx.std(axis=0)
    model = LogisticRegression(C=chosen, class_weight='balanced', max_iter=2000)
    model.fit((tx - mean) / std, ty)
    np.testing.assert_array_equal(result['scores'][~train],
                                  model.decision_function((x[~train] - mean) / std))
