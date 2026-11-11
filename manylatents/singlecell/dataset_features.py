"""Read-only descriptive features of a dataset.

AnnData supplies the uns, obsm, var and layers needed by these descriptions;
preprocessing instead operates on arrays and names. No data is loaded here.
The versioned classifier does not replace preprocessing.looks_like_counts.
"""

import numpy as np
from scipy import sparse

COUNTS_LIKE_VERSION = "counts_like/1"
COUNTS_LIKE_MAX_VALUES = 100_000
COUNTS_LIKE_MIN_VALUES = 1_000
COUNTS_LIKE_SEED = 0
COUNTS_LIKE_TOL = 1e-6
COUNTS_LIKE_PROCESSED_FRACTION = 0.01


def _nonzero_values(matrix):
    if sparse.issparse(matrix):
        # Consolidate duplicate coordinates without changing the caller's storage.
        matrix = matrix.tocsr(copy=True)
        matrix.sum_duplicates()
        values = matrix.data
    else:
        values = np.asarray(matrix)
    return values[values != 0]


def counts_like(matrix) -> str:
    """Classify up to 100,000 nonzero values using counts_like/1.

    Fewer than 1,000 values is undefined. Negative values or more than 1%
    nonintegers are processed; all integers are raw_integer; the rest unknown.
    Sampling is uniform without replacement with seed 0. Nonfinite values
    cannot establish integer counts and yield unknown unless negative.
    """
    values = _nonzero_values(matrix)
    if values.size < COUNTS_LIKE_MIN_VALUES:
        return "undefined"
    if values.size > COUNTS_LIKE_MAX_VALUES:
        indices = np.random.default_rng(COUNTS_LIKE_SEED).choice(
            values.size, COUNTS_LIKE_MAX_VALUES, replace=False
        )
        values = values[indices]
    if np.any(values < 0):
        return "processed"
    if not np.all(np.isfinite(values)):
        return "unknown"
    fraction = np.mean(np.abs(values - np.round(values)) > COUNTS_LIKE_TOL)
    if fraction > COUNTS_LIKE_PROCESSED_FRACTION:
        return "processed"
    return "raw_integer" if fraction == 0 else "unknown"
