"""One matrix of every cached track score per variant, across scorers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from manylatents.dogma import atlas
from manylatents.utils.exceptions import MeasurementUnavailable


@dataclass(frozen=True)
class TrackMatrix:
    """Aligned scores, missing gene-row masks and excluded row counts per scorer."""

    values: np.ndarray
    variant_ids: list[str]
    track_scorer: np.ndarray
    without_rows: dict[str, np.ndarray] = field(default_factory=dict)
    n_dropped: dict[str, int] = field(default_factory=dict)


def load_track_matrix(cache_dirs, scorers, gene_scorers=()) -> TrackMatrix:
    """Concatenate cached scorers into one float64 (variants, tracks) matrix.

    Rows are the intersection of ``scorers``, in the first scorer's order.
    Each ``gene_scorers`` block uses signed max-absolute reduction across genes;
    missing rows get zeros and True in ``without_rows[scorer]``. ``n_dropped``
    counts each scorer's cached variants excluded from the output.

    Missing gene rows are assumed to mean no gene in the input window: callers
    must ensure caches are complete. An entirely absent scorer still raises
    AtlasScoresMissing because its track dimensions are unknown.

    Raises ValueError for empty, overlapping or duplicate scorer requests, and
    MeasurementUnavailable if the ordinary scorers have no common variants.
    """
    scorers, gene_scorers = list(scorers), list(gene_scorers)
    cache_dirs = list(cache_dirs)
    if not scorers:
        raise ValueError("give at least one scorer with one row per variant")
    both = sorted(set(scorers) & set(gene_scorers))
    if both:
        raise ValueError(f"scorers named in both scorers and gene_scorers: {both}")
    if len(set(scorers)) != len(scorers) or len(set(gene_scorers)) != len(gene_scorers):
        raise ValueError("duplicate scorer names")

    loaded = {s: atlas.load_cached_scores(cache_dirs, s) for s in scorers}
    common = set.intersection(*(set(m.variant_ids) for m in loaded.values()))
    order = [v for v in loaded[scorers[0]].variant_ids if v in common]
    if not order:
        raise MeasurementUnavailable("scorers have no common variants")

    blocks, names, without, dropped = [], [], {}, {}
    for scorer in scorers:
        matrix = loaded[scorer]
        position = {v: i for i, v in enumerate(matrix.variant_ids)}
        blocks.append(np.asarray(matrix.values, dtype=np.float64)[[position[v] for v in order]])
        names += [scorer] * matrix.values.shape[1]
        dropped[scorer] = len(matrix.variant_ids) - len(order)
    for scorer in gene_scorers:
        matrix = atlas.load_cached_scores(cache_dirs, scorer, gene_reduce="maxabs")
        position = {v: i for i, v in enumerate(matrix.variant_ids)}
        rows = np.array([position.get(v, -1) for v in order], dtype=int)
        block = np.zeros((len(order), matrix.values.shape[1]), dtype=np.float64)
        block[rows >= 0] = np.asarray(matrix.values, dtype=np.float64)[rows[rows >= 0]]
        blocks.append(block)
        names += [scorer] * matrix.values.shape[1]
        without[scorer] = rows < 0
        dropped[scorer] = len(matrix.variant_ids) - int((rows >= 0).sum())
    return TrackMatrix(values=np.concatenate(blocks, axis=1), variant_ids=order,
                       track_scorer=np.asarray(names), without_rows=without,
                       n_dropped=dropped)
