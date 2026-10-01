"""Synthetic checks for the track geometry evaluation protocol."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest


from manylatents.dogma.benchmarks.track_geometry import (
    Settings, aggregation_scores, evaluate_score, measure_defined,
    neighbour_scores, reference_scores, run, summarize_score, transform_features,
    within_magnitude,
)
from manylatents.utils.exceptions import MeasurementUnavailable


@pytest.fixture
def settings():
    return Settings(primary_k=12, pickands_k=24, k_grid=(10, 12),
                    pca_components=(3, 10), n_bootstrap=5, n_permutations=7,
                    n_control_seeds=2, seed=42,
                    locus_exclusions=(0, 1000, float("inf")))


@pytest.fixture
def data():
    rng = np.random.default_rng(81)
    n, p = 180, 24
    labels = np.tile([True, False, False], n // 3)
    direction = np.tile([1., -1.], p // 2)
    query = direction + rng.normal(scale=0.3, size=(n, p))
    query[labels] += 2 * np.ones(p)
    reference = direction + rng.normal(scale=0.3, size=(220, p))
    # The planted direction changes independently of each row's magnitude.
    query *= rng.uniform(4, 6, size=(n, 1)) / np.linalg.norm(query, axis=1, keepdims=True)
    reference *= rng.uniform(4, 6, size=(220, 1)) / np.linalg.norm(reference, axis=1, keepdims=True)
    variants = pd.DataFrame({"chrom": np.repeat(["1", "2", "3"], n // 3),
                             "pos": rng.integers(1, 2_000_000, size=n), "label": labels})
    return query, {"reference": reference, "transductive": None}, variants


def test_settings_are_frozen():
    settings = Settings(lid_sign=1, pca_components=[2], signs={"lof": -1})
    assert settings.signs["lid"] == 1
    assert settings.signs["pca2_residual"] == 1
    assert settings.signs["lof"] == -1
    assert settings.pca_components == (2,)
    with pytest.raises(FrozenInstanceError):
        settings.seed = 0
    with pytest.raises(TypeError):
        settings.signs["lid"] = -1


@pytest.mark.parametrize("columns", [1, 2])
def test_measure_defined_drops_relative_indices(columns):
    calls = []

    def measurement(keep):
        calls.append(keep.copy())
        for point in (1, 4):
            if point in keep:
                err = MeasurementUnavailable("undefined point")
                err.indices = np.flatnonzero(keep == point)
                raise err
        return keep.astype(float) if columns == 1 else np.column_stack([keep, keep + 10])

    values, dropped, reason = measure_defined(measurement, 6)
    assert dropped == 2 and reason is None
    assert np.isnan(values[[1, 4]]).all()
    expected = np.array([0, 2, 3, 5])
    np.testing.assert_array_equal(values[expected] if columns == 1 else values[expected, 0], expected)
    if columns == 2:
        np.testing.assert_array_equal(values[expected, 1], expected + 10)
    assert len(calls) == 3


def test_measure_undefined_without_indices():
    def measurement(keep):
        raise MeasurementUnavailable("no measurement")

    assert measure_defined(measurement, 6) == (None, 6, "no measurement")


@pytest.mark.parametrize("transductive", [False, True])
def test_transforms(transductive):
    query = np.array([[2., 4.], [5., 9.]])
    reference = None if transductive else np.array([[1., 2.], [3., 8.], [8., 5.]])
    q, r = transform_features(query, reference, "raw")
    assert q is query and r is reference
    for how in ("unit", "log", "zscore"):
        q, r = transform_features(query, reference, how)
        cloud = query if reference is None else reference
        if how == "unit":
            np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1)
            if r is not None:
                np.testing.assert_allclose(np.linalg.norm(r, axis=1), 1)
        elif how == "log":
            np.testing.assert_array_equal(q, np.log(query))
            if r is not None:
                np.testing.assert_array_equal(r, np.log(reference))
        else:
            np.testing.assert_allclose(q, (query - cloud.mean(axis=0)) / cloud.std(axis=0))
            if r is not None:
                np.testing.assert_allclose(r, (cloud - cloud.mean(axis=0)) / cloud.std(axis=0))
        if transductive:
            assert r is None
    with pytest.raises(MeasurementUnavailable, match="zero norm"):
        transform_features(np.zeros((1, 2)), reference, "unit")
    with pytest.raises(MeasurementUnavailable, match="zero norm"):
        transform_features(query, np.zeros((1, 2)), "unit")
    with pytest.raises(ValueError):
        transform_features(query, reference, "unknown")


@pytest.mark.parametrize("cloud", ["reference", "transductive"])
def test_reference_scores(data, settings, cloud):
    query, clouds, _ = data
    scores, dropped = reference_scores(query, clouds[cloud], 12, 24, settings=settings)
    assert set(scores) == {"lid", "exponentiality", "knn_distance", "pickands_xi",
                           "participation_ratio", "lof", "pca3_leading", "pca3_residual",
                           "pca10_leading", "pca10_residual"}
    assert not dropped
    assert all(v.shape == (len(query),) and np.isfinite(v).all() for v in scores.values())
    for only, expected in [(('lof',), {'lof'}),
                           (('lid',), {'lid', 'exponentiality', 'knn_distance'}),
                           (('pca3_residual',), {'pca3_leading', 'pca3_residual'})]:
        subset, _ = reference_scores(query, clouds[cloud], 12, 24, only=only, settings=settings)
        assert set(subset) == expected
        for name in subset:
            np.testing.assert_array_equal(subset[name], scores[name])


def test_evaluation(data, settings):
    query, _, variants = data
    labels, groups = variants.label.to_numpy(), variants.chrom.to_numpy()
    baselines = aggregation_scores(query)
    covariates = np.column_stack([baselines['l2'], baselines['max']])
    values = labels + np.random.default_rng(42).normal(0, .01, len(labels))
    for name in ('lof', 'l2', 'max'):
        summary = summarize_score(name, values, labels, groups, covariates)
        assert summary['auprc_plus'] > .95
        assert summary['auprc_minus'] < .4
        assert ('partial_rho_given_l2_max' in summary) == (name == 'lof')
        evaluated = evaluate_score(name, values, labels, groups, covariates,
                                   np.random.default_rng(42), settings=settings)
        assert evaluated.items() >= summary.items()
        assert evaluated['auprc'] == summary['auprc_plus']
        assert np.isfinite([evaluated['se'], evaluated['se_groups']]).all()
    unavailable = values.copy()
    unavailable[:100] = np.nan
    for fn in (summarize_score, evaluate_score):
        kwargs = {} if fn is summarize_score else {'rng': np.random.default_rng(1), 'settings': settings}
        result = fn('lof', unavailable, labels, groups, covariates, **kwargs)
        assert result['unavailable'] is True and result['n_dropped'] == 100
    bins = within_magnitude(values, labels, baselines['l2'], 1)
    assert sum(b['n'] for b in bins) == len(labels)
    assert all(b['ap'] == b['ap_plus'] for b in bins)


def test_neighbour_scores():
    scores = neighbour_scores(np.array([[1., 2., 3.], [0., 1., 2.], [1., 2., np.inf]]))
    assert np.isfinite(scores['lid'][0])
    assert scores['knn_distance'][0] == 2
    assert all(np.isnan(v[1:]).all() for v in scores.values())


@pytest.mark.parametrize('use_random_tracks', [False, True])
def test_run(data, settings, use_random_tracks):
    query, clouds, variants = data
    calls, messages = [], []

    def random_tracks(rng):
        calls.append(True)
        return (rng.normal(size=query.shape),
                {name: None if cloud is None else rng.normal(size=cloud.shape)
                 for name, cloud in clouds.items()})

    def execute():
        return run(query=query, clouds=clouds, variants=variants, primary_cloud='reference',
                   transforms=('raw', 'zscore', 'unit'),
                   control_settings=(('reference', 'raw', 12), ('transductive', 'unit', 12)),
                   random_tracks=random_tracks if use_random_tracks else None,
                   rng=np.random.default_rng(settings.seed), report={'metadata': 'retained'},
                   settings=settings, log=messages.append)

    report, scores = execute()
    repeated, repeated_scores = execute()
    assert set(report) == {'metadata', 'n_variants', 'n_positive', 'n_tracks', 'clouds',
                           'primary', 'prevalence', 'dropped', 'scores', 'within_l2_quintile',
                           'permutation_null', 'sensitivity', 'controls', 'locus', 'seconds'}
    assert report['metadata'] == 'retained'
    report.pop('seconds')
    repeated.pop('seconds')
    assert report == repeated
    for name in scores:
        np.testing.assert_array_equal(scores[name], repeated_scores[name])
    controls = report['controls']['reference|raw|k=12']
    assert any(report['scores'][name][f'auprc_{sign}'] > controls[name]['row_shuffle'][sign]['mean'] + .1
               for name in ('lid', 'lof', 'participation_ratio') for sign in ('plus', 'minus'))
    assert len(calls) == (2 * 2 * settings.n_control_seeds if use_random_tracks else 0)
    assert all(('random_tracks' in kinds) == use_random_tracks for kinds in controls.values())
    assert any(m.startswith('sensitivity grid: reference done') for m in messages)
    assert any(m.startswith('controls: reference|raw|k=12 done') for m in messages)
    assert any(m.startswith('locus analysis done') for m in messages)
