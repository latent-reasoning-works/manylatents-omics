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


DEFAULT_C_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def chromosome_split_probe(features, labels, chroms, *, c_grid=DEFAULT_C_GRID,
                           train=AVI_TRAIN_CHROMOSOMES, tune=AVI_TUNE_CHROMOSOMES,
                           max_iter=2000) -> dict:
    """L2 logistic regression fitted on the training chromosomes.

    Features are standardised with training moments, dropping columns whose
    standard deviation is below 1e-12 times max(1, largest absolute train value).
    Classes are balanced; best pooled tuning average precision selects C, with
    ties going to the smaller C. Report rows never enter fitting or tuning.

    Returns ``scores`` for every row, ``split``, ``chosen_c``, ``tune_auprc``
    (C to pooled AP), ``chosen_c_on_edge`` (endpoint of a multi-value grid),
    ``n_features_used``, ``n_features_dropped``,
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

    train_labels = labels[in_train].astype(bool)
    tune_labels = labels[in_tune].astype(bool)
    scaled, kept = _standardize_training(features, features[in_train])

    tuning, best = {}, None
    for c in grid:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(scaled[in_train], train_labels)
        except ConvergenceWarning as error:
            raise MeasurementUnavailable(f"probe did not converge for C={c}") from error
        value = float(average_precision_score(tune_labels, _decision_scores(model, scaled[in_tune])))
        tuning[c] = value
        if best is None or value > best[0]:
            best = (value, c, model)
    _, chosen, model = best
    return {
        "scores": _decision_scores(model, scaled),
        "split": split,
        "chosen_c": chosen,
        "chosen_c_on_edge": len(grid) > 1 and chosen in (grid[0], grid[-1]),
        "tune_auprc": tuning,
        "n_features_used": int(kept.sum()),
        "n_features_dropped": int((~kept).sum()),
        "kept_features": kept,
        "n_train": int(in_train.sum()),
        "n_train_positive": int(train_labels.sum()),
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


def leave_chromosome_out_probe(features, labels, chroms, *, c_grid=DEFAULT_C_GRID,
                               inner_folds=5, max_iter=2000, n_jobs=1) -> dict:
    """Out-of-fold logistic decision values with nested chromosome-group tuning.

    Each chromosome is scored by a model fitted on all other chromosomes. C is
    selected by pooled average precision over grouped inner validation folds;
    ties choose the smaller C. Every fit standardises from its own training
    rows and drops near-constant columns using the split probe's tolerance.
    Single-class held-out chromosomes are supported. Inner folds without both
    training classes are skipped and counted, including their excluded rows.

    Returns scores in input order, per-chromosome dictionaries ``chosen_c``,
    ``chosen_c_on_edge``, ``n_features_used``, ``n_features_dropped``,
    ``n_inner_folds_skipped``, ``n_inner_rows_skipped``, and ``n_chromosomes``.
    Chromosome keys are normalised. ``n_jobs`` parallelises outer folds only.

    Raises ValueError for malformed inputs or parameters, MeasurementUnavailable
    for nonfinite data, nonbinary labels, undefined fits or tuning AP, no usable
    inner folds, nonfinite predictions, or failed convergence.
    """
    from numbers import Integral

    from manylatents.dogma.benchmarks.splits import normalize_chromosomes

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    chroms = np.asarray(chroms, dtype=object)
    if (features.ndim != 2 or labels.ndim != 1 or chroms.ndim != 1
            or not (len(features) == len(labels) == len(chroms))):
        raise ValueError("features (2-D), labels and chroms (1-D) must have the same length")
    chroms = normalize_chromosomes(chroms)
    if not np.isfinite(features).all():
        raise MeasurementUnavailable("features are not all finite")
    if not np.isin(labels, [False, True]).all():
        raise MeasurementUnavailable("labels must be binary (0 or 1)")
    labels = labels.astype(bool)
    grid = sorted(set(float(c) for c in c_grid))
    if not grid or not all(np.isfinite(c) and c > 0 for c in grid):
        raise ValueError("c_grid must contain finite positive values")
    if isinstance(inner_folds, bool) or not isinstance(inner_folds, Integral) or inner_folds < 2:
        raise ValueError("inner_folds must be an integer of at least two")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")
    chromosomes = np.unique(chroms)
    if len(chromosomes) < 3:
        raise MeasurementUnavailable("nested tuning needs at least three chromosomes")
    arguments = (features, labels, chroms, grid, inner_folds, max_iter)
    if n_jobs == 1:
        folds = [_leave_chromosome_out(chrom, *arguments) for chrom in chromosomes]
    else:
        from joblib import Parallel, delayed

        folds = Parallel(n_jobs=n_jobs)(
            delayed(_leave_chromosome_out)(chrom, *arguments) for chrom in chromosomes
        )
    scores = np.empty(len(features), dtype=float)
    result = {key: {} for key in (
        "chosen_c", "chosen_c_on_edge", "n_features_used", "n_features_dropped",
        "n_inner_folds_skipped", "n_inner_rows_skipped",
    )}
    for chrom, (fold_scores, details) in zip(chromosomes, folds):
        scores[chroms == chrom] = fold_scores
        for key, value in details.items():
            result[key][str(chrom)] = value
    return {"scores": scores, **result, "n_chromosomes": len(chromosomes)}


def _fit_probe(features, labels, c, max_iter):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(features, labels)
    except ConvergenceWarning as error:
        raise MeasurementUnavailable(f"probe did not converge for C={c}") from error
    return model


def _leave_chromosome_out(chrom, features, labels, chroms, grid, inner_folds, max_iter):
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold

    train = chroms != chrom
    x, y, groups = features[train], labels[train], chroms[train]
    if np.unique(y).size != 2:
        raise MeasurementUnavailable(f"training chromosomes for {chrom} do not hold both classes")
    splitter = GroupKFold(n_splits=min(inner_folds, len(np.unique(groups))))
    folds, skipped, skipped_rows = [], 0, 0
    for fit, valid in splitter.split(x, y, groups):
        if np.unique(y[fit]).size != 2:
            skipped += 1
            skipped_rows += len(valid)
            continue
        scaled, _ = _standardize_training(x, x[fit])
        folds.append((scaled[fit], y[fit], scaled[valid], y[valid]))
    if not folds:
        raise MeasurementUnavailable(f"no usable inner folds for chromosome {chrom}")
    pooled_labels = np.concatenate([fold[3] for fold in folds])
    if not pooled_labels.any():
        raise MeasurementUnavailable(f"inner validation for {chrom} has no positives")
    best_ap, chosen = -np.inf, None
    for c in grid:
        decisions = []
        for fit_x, fit_y, valid_x, _ in folds:
            model = _fit_probe(fit_x, fit_y, c, max_iter)
            decisions.append(_decision_scores(model, valid_x))
        ap = float(average_precision_score(pooled_labels, np.concatenate(decisions)))
        if ap > best_ap:
            best_ap, chosen = ap, c
    scaled, kept = _standardize_training(features, x)
    model = _fit_probe(scaled[train], y, chosen, max_iter)
    return _decision_scores(model, scaled[~train]), {
        "chosen_c": chosen,
        "chosen_c_on_edge": len(grid) > 1 and chosen in (grid[0], grid[-1]),
        "n_features_used": int(kept.sum()),
        "n_features_dropped": int((~kept).sum()),
        "n_inner_folds_skipped": skipped,
        "n_inner_rows_skipped": skipped_rows,
    }
