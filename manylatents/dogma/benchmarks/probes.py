"""Linear probes of variant features, trained and tuned on separate chromosomes."""
from __future__ import annotations

import warnings

import numpy as np

from manylatents.dogma.benchmarks.splits import (
    AVI_TRAIN_CHROMOSOMES,
    AVI_TUNE_CHROMOSOMES,
    chromosome_split,
)
from manylatents.utils.exceptions import MeasurementUnavailable


def chromosome_split_probe(features, labels, chroms, *, c_grid=(1e-3, 1e-2, 1e-1, 1.0),
                           train=AVI_TRAIN_CHROMOSOMES, tune=AVI_TUNE_CHROMOSOMES,
                           max_iter=2000) -> dict:
    """L2 logistic regression fitted on the training chromosomes.

    Features are standardised with training moments, dropping constant columns.
    Classes are balanced; best pooled tuning average precision selects C, with
    ties going to the smaller C. Report rows never enter fitting or tuning.

    Returns ``scores`` for every row, ``split``, ``chosen_c``, ``tune_auprc``
    (C to pooled AP), ``n_features_used``, ``n_features_dropped``,
    ``kept_features`` (column mask), ``n_train`` and ``n_train_positive``.

    Raises ValueError for incompatible shapes or invalid C grids, and
    MeasurementUnavailable for nonfinite features, train/tune without both
    classes, invalid binary labels, no varying training features, nonfinite
    predictions, or a fit that does not converge.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    chroms = np.asarray(chroms, dtype=object)
    if (features.ndim != 2 or labels.ndim != 1 or chroms.ndim != 1
            or not (len(features) == len(labels) == len(chroms))):
        raise ValueError("features (2-D), labels and chroms (1-D) must have the same length")
    if not np.isfinite(features).all():
        raise MeasurementUnavailable("features are not all finite")
    grid = sorted(set(float(c) for c in c_grid))
    if not grid or not all(np.isfinite(c) and c > 0 for c in grid):
        raise ValueError("c_grid must contain finite positive values")
    split = chromosome_split(chroms, train, tune)
    in_train, in_tune = split == "train", split == "tune"
    for name, rows in (("train", in_train), ("tune", in_tune)):
        if not np.isin(labels[rows], [False, True]).all():
            raise MeasurementUnavailable(f"{name} labels must be binary (0 or 1)")
        if np.unique(labels[rows]).size < 2:
            raise MeasurementUnavailable(f"{name} chromosomes do not hold both classes")

    scaled, kept = _standardize_training(features, features[in_train])

    tuning, best = {}, None
    for c in grid:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(scaled[in_train], labels[in_train])
        except ConvergenceWarning as error:
            raise MeasurementUnavailable(f"probe did not converge for C={c}") from error
        value = float(average_precision_score(labels[in_tune], _decision_scores(model, scaled[in_tune])))
        tuning[c] = value
        if best is None or value > best[0]:
            best = (value, c, model)
    _, chosen, model = best
    return {
        "scores": _decision_scores(model, scaled),
        "split": split,
        "chosen_c": chosen,
        "tune_auprc": tuning,
        "n_features_used": int(kept.sum()),
        "n_features_dropped": int((~kept).sum()),
        "kept_features": kept,
        "n_train": int(in_train.sum()),
        "n_train_positive": int(labels[in_train].sum()),
    }


def _decision_scores(model, features):
    scores = model.decision_function(features)
    if not np.isfinite(scores).all():
        raise MeasurementUnavailable("probe decision values are not finite")
    return scores


def _standardize_training(features, reference):
    """Scale using only reference rows, counting numerically constant columns."""
    mean, spread = reference.mean(axis=0), reference.std(axis=0)
    tolerance = 1e-12 * np.maximum(1.0, np.abs(reference).max(axis=0))
    if not (np.isfinite(mean).all() and np.isfinite(spread).all()):
        raise MeasurementUnavailable("training moments are not finite")
    kept = spread >= tolerance
    if not kept.any():
        raise MeasurementUnavailable("no feature varies on the training chromosomes")
    scaled = (features[:, kept] - mean[kept]) / spread[kept]
    if not np.isfinite(scaled).all():
        raise MeasurementUnavailable("standardised features are not finite")
    return scaled, kept
