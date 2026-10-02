"""Paired uncertainty intervals for unmatched variant benchmarks."""
from numbers import Integral

import numpy as np

from manylatents.utils.exceptions import MeasurementUnavailable


def clustered_average_precision_difference(scores_a, scores_b, labels, clusters, *,
                                           n_bootstrap=1000, rng) -> dict:
    """Pooled AP difference with a paired, class-stratified cluster bootstrap.

    Within each class, draw as many clusters as observed, with replacement,
    retaining every row of that class in each drawn cluster. Mixed-class
    clusters participate separately in the two draws. Both score vectors use
    exactly the same sampled rows. ``rng`` must be a NumPy Generator.

    Returns ``auprc_a``, ``auprc_b``, ``difference``, bootstrap sample ``se``
    (ddof=1), percentile ``ci95``, and the cluster counts for each class.
    Raises ValueError for incompatible vectors or fewer than two bootstrap
    draws, TypeError for an invalid generator, and MeasurementUnavailable for
    nonfinite scores, nonbinary or single-class labels, missing cluster IDs,
    or fewer than two clusters in either class.
    """
    from sklearn.metrics import average_precision_score

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least two")
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    y, clusters = np.asarray(labels), np.asarray(clusters)
    if (any(v.ndim != 1 for v in (a, b, y, clusters))
            or not len(a) == len(b) == len(y) == len(clusters)):
        raise ValueError("scores, labels and clusters must be vectors of the same length")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise MeasurementUnavailable("scores are not all finite")
    if not np.isin(y, [False, True]).all():
        raise MeasurementUnavailable("labels must be binary (0 or 1)")
    y = y.astype(bool)
    if np.unique(y).size != 2:
        raise MeasurementUnavailable("labels must hold both classes")
    groups = []
    for label in (True, False):
        members = {}
        for row in np.flatnonzero(y == label):
            cluster = clusters[row]
            if cluster is None or cluster != cluster:
                raise MeasurementUnavailable("cluster identifiers must not be missing")
            members.setdefault(cluster, []).append(row)
        # Sorted IDs make draw ordering independent of row order for normal
        # numeric/string cluster identifiers.
        members = [np.asarray(members[key], dtype=int) for key in sorted(members)]
        if len(members) < 2:
            raise MeasurementUnavailable(f"fewer than two clusters in class {int(label)}")
        groups.append(members)
    ap_a, ap_b = float(average_precision_score(y, a)), float(average_precision_score(y, b))
    differences = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        rows = np.concatenate([
            members[index] for members in groups
            for index in rng.choice(len(members), size=len(members), replace=True)
        ])
        differences[iteration] = (average_precision_score(y[rows], a[rows])
                                  - average_precision_score(y[rows], b[rows]))
    return {
        "auprc_a": ap_a,
        "auprc_b": ap_b,
        "difference": ap_a - ap_b,
        "se": float(differences.std(ddof=1)),
        "ci95": tuple(float(x) for x in np.percentile(differences, [2.5, 97.5])),
        "n_clusters_positive": len(groups[0]),
        "n_clusters_negative": len(groups[1]),
    }
