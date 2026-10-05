"""Linear probe with a chromosome split."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.metrics import average_precision_score  # noqa: E402

from manylatents.dogma.benchmarks.probes import chromosome_split_probe  # noqa: E402
from manylatents.utils.exceptions import MeasurementUnavailable  # noqa: E402


def _toy(seed=0, per_chromosome=80):
    rng = np.random.default_rng(seed)
    chroms = np.repeat(["1", "4", "2", "5", "3", "6"], per_chromosome)
    labels = rng.random(len(chroms)) < 0.3
    features = rng.normal(size=(len(chroms), 5))
    features[:, 0] += 2.5 * labels
    return features, labels, chroms


def test_informative_feature_is_found_on_report_chromosomes():
    features, labels, chroms = _toy()
    result = chromosome_split_probe(features, labels, chroms)
    report = result["split"] == "report"
    assert report.sum() == 160
    assert average_precision_score(labels[report], result["scores"][report]) > 0.8
    assert result["chosen_c"] in result["tune_auprc"]
    assert set(result["tune_auprc"]) == {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}
    assert result["n_features_used"] == 5
    assert result["n_train"] == 160 and result["n_train_positive"] == int(labels[:160].sum())


def test_report_rows_do_not_influence_the_fit():
    features, labels, chroms = _toy()
    base = chromosome_split_probe(features, labels, chroms)
    report = base["split"] == "report"
    flipped = labels.copy()
    flipped[report] = ~flipped[report]
    scaled = features.copy()
    scaled[report] *= 1000.0
    other = chromosome_split_probe(scaled, flipped, chroms)
    np.testing.assert_array_equal(other["scores"][~report], base["scores"][~report])
    assert other["chosen_c"] == base["chosen_c"]


def test_feature_constant_on_training_chromosomes_is_dropped():
    features, labels, chroms = _toy()
    features[:160, 1] = 3.0
    result = chromosome_split_probe(features, labels, chroms)
    assert result["n_features_used"] == 4
    assert np.isfinite(result["scores"]).all()


def test_undefined_fits_are_refused():
    features, labels, chroms = _toy()
    one_class = labels.copy()
    one_class[:160] = False
    with pytest.raises(MeasurementUnavailable, match="train"):
        chromosome_split_probe(features, one_class, chroms)
    no_tune = np.where(np.isin(chroms, ["2", "5"]), "3", chroms)
    with pytest.raises(MeasurementUnavailable, match="tune"):
        chromosome_split_probe(features, labels, no_tune)
    broken = features.copy()
    broken[0, 0] = np.nan
    with pytest.raises(MeasurementUnavailable, match="finite"):
        chromosome_split_probe(broken, labels, chroms)
    with pytest.raises(ValueError, match="same length"):
        chromosome_split_probe(features, labels[:-1], chroms)


def test_report_labels_alone_do_not_change_any_score():
    features, labels, chroms = _toy()
    base = chromosome_split_probe(features, labels, chroms)
    labels[base["split"] == "report"] = ~labels[base["split"] == "report"]
    other = chromosome_split_probe(features, labels, chroms)
    np.testing.assert_array_equal(base["scores"], other["scores"])
    assert base["tune_auprc"] == other["tune_auprc"]


def test_one_report_outlier_does_not_change_other_report_predictions():
    features, labels, chroms = _toy()
    base = chromosome_split_probe(features, labels, chroms)
    features[-1] *= 10000
    other = chromosome_split_probe(features, labels, chroms)
    np.testing.assert_array_equal(base["scores"][:-1], other["scores"][:-1])
    assert base["tune_auprc"] == other["tune_auprc"]


def test_decimal_constant_is_dropped_and_its_report_values_are_ignored():
    features, labels, chroms = _toy()
    features[:160, 1] = 0.1
    base = chromosome_split_probe(features, labels, chroms)
    assert base["n_features_used"] == 4
    assert base["n_features_dropped"] == 1
    assert base["kept_features"].tolist() == [True, False, True, True, True]
    features[160:, 1] = 1e6
    other = chromosome_split_probe(features, labels, chroms)
    np.testing.assert_array_equal(base["scores"], other["scores"])


def test_all_constant_training_features_are_undefined():
    features, labels, chroms = _toy()
    features[:160] = 0.1
    with pytest.raises(MeasurementUnavailable, match="train"):
        chromosome_split_probe(features, labels, chroms)


def test_tuning_requires_both_classes():
    features, labels, chroms = _toy()
    labels[160:320] = True
    with pytest.raises(MeasurementUnavailable, match="tune"):
        chromosome_split_probe(features, labels, chroms)


def test_prefixes_do_not_change_fit_or_split():
    features, labels, chroms = _toy()
    base = chromosome_split_probe(features, labels, chroms)
    prefixed = [f"chr{c}" if i % 2 else c for i, c in enumerate(chroms)]
    other = chromosome_split_probe(features, labels, prefixed)
    np.testing.assert_array_equal(base["scores"], other["scores"])
    np.testing.assert_array_equal(base["split"], other["split"])


def test_ties_choose_smallest_c_independent_of_grid_order():
    features = np.tile([[-1.], [1.]], (6, 1))
    labels = np.tile([False, True], 6)
    chroms = np.repeat(["1", "2", "3"], 4)
    result = chromosome_split_probe(features, labels, chroms, c_grid=[10, 1, 0.01])
    assert result["tune_auprc"] == {0.01: 1.0, 1.0: 1.0, 10.0: 1.0}
    assert result["chosen_c"] == 0.01


@pytest.mark.parametrize("grid", [[], [0], [-1], [np.nan], [np.inf]])
def test_invalid_c_grid_is_refused(grid):
    with pytest.raises(ValueError, match="c_grid"):
        chromosome_split_probe(*_toy(), c_grid=grid)


@pytest.mark.parametrize("which", ["labels", "chroms"])
def test_nonvector_inputs_are_refused(which):
    features, labels, chroms = _toy()
    if which == "labels":
        labels = labels[:, None]
    else:
        chroms = chroms[:, None]
    with pytest.raises(ValueError, match="same length"):
        chromosome_split_probe(features, labels, chroms)


def test_unconverged_fit_is_unavailable():
    with pytest.raises(MeasurementUnavailable, match="converge"):
        chromosome_split_probe(*_toy(), max_iter=1)


@pytest.mark.parametrize("row,value", [(0, np.nan), (160, 2)])
def test_invalid_training_or_tuning_labels_are_refused(row, value):
    features, labels, chroms = _toy()
    labels = labels.astype(float)
    labels[row] = value
    with pytest.raises(MeasurementUnavailable, match="labels"):
        chromosome_split_probe(features, labels, chroms)


def test_nonfinite_decision_values_are_unavailable():
    features = np.array([[-1.], [1.], [-1.], [1.], [1e308]])
    labels = [False, True, False, True, False]
    chroms = ["1", "1", "2", "2", "3"]
    with pytest.raises(MeasurementUnavailable, match="finite"):
        chromosome_split_probe(features, labels, chroms, c_grid=[1e6])


def test_near_constant_training_feature_is_dropped_and_counted():
    features, labels, chroms = _toy()
    features[:160, 1] = 0
    features[0, 1] = 1e-12
    result = chromosome_split_probe(features, labels, chroms)
    assert result['n_features_used'] == 4
    assert result['n_features_dropped'] == 1
    features[160:, 1] *= 1e9
    other = chromosome_split_probe(features, labels, chroms)
    np.testing.assert_array_equal(result['scores'], other['scores'])


def test_object_boolean_labels_match_boolean_scores_and_ignore_report_labels():
    features, labels, chroms = _toy()
    base = chromosome_split_probe(features, labels, chroms)
    labels = labels.astype(object)
    labels[320:] = None
    other = chromosome_split_probe(features, labels, chroms)
    np.testing.assert_array_equal(base['scores'], other['scores'])


@pytest.mark.parametrize('grid,edge', [([1], False), ([1, 1], False), ([0.01, 1], True)])
def test_chosen_c_grid_edge_is_explicit(grid, edge):
    features = np.tile([[-1.], [1.]], (6, 1))
    labels = np.tile([False, True], 6)
    chroms = np.repeat(['1', '2', '3'], 4)
    result = chromosome_split_probe(features, labels, chroms, c_grid=grid)
    assert result['chosen_c_on_edge'] is edge


@pytest.mark.parametrize('values,chosen,gain', [
    ([0.8, 0.6, 0.5], 0.01, 0.2),
    ([0.5, 0.8, 0.6], 1.0, 0.0),
    ([0.5, 0.6, 0.8], 10.0, 0.2),
    ([0.8, 0.8, 0.8], 0.01, 0.0),
    ([0.8], 0.01, 0.0),
])
def test_nested_tuning_curve_and_edge_gain(monkeypatch, values, chosen, gain):
    from manylatents.dogma.benchmarks.probes import leave_chromosome_out_probe
    grid = [0.01, 1.0, 10.0][:len(values)]
    calls = []

    def pooled_ap(y, scores):
        calls.append((len(y), len(scores)))
        return values[(len(calls) - 1) % len(values)]

    monkeypatch.setattr('sklearn.metrics.average_precision_score', pooled_ap)
    features, labels, chroms = _toy(per_chromosome=12)
    result = leave_chromosome_out_probe(features, labels, chroms,
                                        c_grid=grid[::-1], inner_folds=2)
    assert calls == [(60, 60)] * (6 * len(grid))
    for chrom in np.unique(chroms):
        assert result['tune_auprc'][chrom] == dict(zip(grid, values))
        assert result['chosen_c'][chrom] == chosen
        assert result['chosen_c_edge_gain'][chrom] == pytest.approx(gain)


def test_nested_weights_reconstruct_fold_decisions():
    from manylatents.dogma.benchmarks.probes import leave_chromosome_out_probe, _standardize_training
    x, y, chroms = _toy(per_chromosome=12)
    x[:, -1] = 3
    result = leave_chromosome_out_probe(x, y, chroms, c_grid=[.1], inner_folds=2)
    for chrom in np.unique(chroms):
        scaled, kept = _standardize_training(x, x[chroms != chrom])
        weights = np.asarray(result['standardised_weights'][chrom])
        assert weights.shape == (x.shape[1],)
        assert weights[-1] == 0
        predicted = scaled[chroms == chrom] @ weights[kept] + result['intercept'][chrom]
        np.testing.assert_allclose(predicted, result['scores'][chroms == chrom])
