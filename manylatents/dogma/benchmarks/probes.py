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
    ``chosen_c_on_edge``, ``tune_auprc`` (C to pooled inner-fold AP),
    ``chosen_c_edge_gain`` (chosen AP minus its sole grid neighbour at an
    endpoint, zero for interior choices or singleton grids),
    ``n_features_used``, ``n_features_dropped``,
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
        "n_inner_folds_skipped", "n_inner_rows_skipped", "tune_auprc", "chosen_c_edge_gain",
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
    tuning = {}
    best_ap, chosen = -np.inf, None
    for c in grid:
        decisions = []
        for fit_x, fit_y, valid_x, _ in folds:
            model = _fit_probe(fit_x, fit_y, c, max_iter)
            decisions.append(_decision_scores(model, valid_x))
        ap = float(average_precision_score(pooled_labels, np.concatenate(decisions)))
        tuning[c] = ap
        if ap > best_ap:
            best_ap, chosen = ap, c
    edge_gain = 0.0
    if len(grid) > 1 and chosen in (grid[0], grid[-1]):
        neighbour = grid[1] if chosen == grid[0] else grid[-2]
        edge_gain = tuning[chosen] - tuning[neighbour]
    scaled, kept = _standardize_training(features, x)
    model = _fit_probe(scaled[train], y, chosen, max_iter)
    return _decision_scores(model, scaled[~train]), {
        "chosen_c": chosen,
        "chosen_c_on_edge": len(grid) > 1 and chosen in (grid[0], grid[-1]),
        "n_features_used": int(kept.sum()),
        "n_features_dropped": int((~kept).sum()),
        "tune_auprc": tuning,
        "chosen_c_edge_gain": edge_gain,
        "n_inner_folds_skipped": skipped,
        "n_inner_rows_skipped": skipped_rows,
    }


DEFAULT_TREE_GRID = ((4, 100), (4, 300), (16, 100), (16, 300))


def leave_chromosome_out_tree_probe(features, labels, chroms, *, grid=DEFAULT_TREE_GRID,
                                    inner_folds=3, n_jobs=1, random_state):
    """Out-of-fold boosted-tree decision values with grouped inner tuning.

    ``grid`` contains (max_leaf_nodes, max_iter) pairs. Pooled inner average
    precision selects the configuration; ties prefer fewer leaves, then fewer
    iterations. Fixed settings are learning_rate=.05, min_samples_leaf=20,
    l2_regularization=1, balanced classes and no early stopping. No scaling or
    feature selection is done. The grid has no edge-failure criterion.

    Like the linear probe, single-class inner training folds are skipped and
    counted. Returns input-order ``scores``, ``chosen_config`` and
    ``tune_auprc`` per chromosome, feature counts, skipped-fold/row counts and
    ``n_chromosomes``. Tuning keys name both configuration parameters.
    Outer folds alone are parallelised. Each fit uses one native thread for
    deterministic predictions independent of ``n_jobs``.

    Raises ValueError for malformed inputs/settings and MeasurementUnavailable
    for nonfinite data, nonbinary labels or undefined training/tuning.
    """
    from numbers import Integral
    from manylatents.dogma.benchmarks.splits import normalize_chromosomes

    features = np.asarray(features, dtype=np.float64)
    labels, chroms = np.asarray(labels), np.asarray(chroms, dtype=object)
    if (features.ndim != 2 or labels.ndim != 1 or chroms.ndim != 1
            or not len(features) == len(labels) == len(chroms)):
        raise ValueError("features (2-D), labels and chroms (1-D) must have the same length")
    chroms = normalize_chromosomes(chroms)
    if not np.isfinite(features).all():
        raise MeasurementUnavailable("features are not all finite")
    if not np.isin(labels, [False, True]).all():
        raise MeasurementUnavailable("labels must be binary (0 or 1)")
    if not features.shape[1]:
        raise MeasurementUnavailable("no input features")
    pairs = []
    for config in grid:
        if (not isinstance(config, (tuple, list)) or len(config) != 2
                or any(isinstance(v, bool) or not isinstance(v, Integral) for v in config)
                or config[0] < 2 or config[1] < 1):
            raise ValueError("grid must contain integer (max_leaf_nodes >= 2, max_iter >= 1) pairs")
        pairs.append(tuple(map(int, config)))
    grid = sorted(set(pairs))
    if not grid:
        raise ValueError("grid must not be empty")
    for name, value, minimum in (("inner_folds", inner_folds, 2), ("random_state", random_state, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if random_state > 2**32 - 1:
        raise ValueError("random_state must be less than 2**32")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")
    chromosomes = np.unique(chroms)
    if len(chromosomes) < 3:
        raise MeasurementUnavailable("nested tuning needs at least three chromosomes")
    args = (features, labels.astype(bool), chroms, grid, inner_folds, random_state)
    if n_jobs == 1:
        folds = [_leave_chromosome_out_tree(c, *args) for c in chromosomes]
    else:
        from joblib import Parallel, delayed
        folds = Parallel(n_jobs=n_jobs)(delayed(_leave_chromosome_out_tree)(c, *args) for c in chromosomes)
    scores, result = np.empty(len(features)), {}
    for chrom, (predictions, details) in zip(chromosomes, folds):
        scores[chroms == chrom] = predictions
        for key, value in details.items():
            result.setdefault(key, {})[str(chrom)] = value
    return {"scores": scores, **result, "n_chromosomes": len(chromosomes)}


def _leave_chromosome_out_tree(chrom, features, labels, chroms, grid, inner_folds, random_state):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits

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
        else:
            folds.append((fit, valid))
    if not folds:
        raise MeasurementUnavailable(f"no usable inner folds for chromosome {chrom}")
    pooled_labels = np.concatenate([y[valid] for _, valid in folds])
    if not pooled_labels.any():
        raise MeasurementUnavailable(f"inner validation for {chrom} has no positives")

    def model(config):
        return HistGradientBoostingClassifier(
            max_leaf_nodes=config[0], max_iter=config[1], learning_rate=.05,
            min_samples_leaf=20, l2_regularization=1., class_weight="balanced",
            early_stopping=False, random_state=int(random_state))

    tuning, best_ap, chosen = {}, -np.inf, None
    with threadpool_limits(limits=1):
        for config in grid:
            predictions = [_decision_scores(model(config).fit(x[fit], y[fit]), x[valid])
                           for fit, valid in folds]
            ap = float(average_precision_score(pooled_labels, np.concatenate(predictions)))
            tuning[f"max_leaf_nodes={config[0]},max_iter={config[1]}"] = ap
            if ap > best_ap:
                best_ap, chosen = ap, config
        scores = _decision_scores(model(chosen).fit(x, y), features[~train])
    return scores, {
        "chosen_config": {"max_leaf_nodes": chosen[0], "max_iter": chosen[1]},
        "tune_auprc": tuning,
        "n_features_used": features.shape[1], "n_features_dropped": 0,
        "n_inner_folds_skipped": skipped, "n_inner_rows_skipped": skipped_rows,
    }
