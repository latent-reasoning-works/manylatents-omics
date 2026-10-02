"""Reductions of a (variants, tracks) score matrix used by published methods."""
from __future__ import annotations

from numbers import Integral

import numpy as np

from manylatents.utils.exceptions import MeasurementUnavailable

# AlphaGenome Variant Impact: splicing sums the maxima of its three scorers.
AVI_MODALITIES = {
    "DNASE": ("DNASE",),
    "ATAC": ("ATAC",),
    "CHIP_TF": ("CHIP_TF",),
    "CHIP_HISTONE": ("CHIP_HISTONE",),
    "CAGE": ("CAGE",),
    "PROCAP": ("PROCAP",),
    "RNA_SEQ": ("RNA_SEQ",),
    "POLYADENYLATION": ("POLYADENYLATION",),
    "SPLICING": ("SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS"),
    "CONTACT_MAPS": ("CONTACT_MAPS",),
}


class FeatureReduction(tuple):
    """Feature arrays and names, unpackable as a tuple, with ``n_dropped`` counts.

    ``n_dropped`` maps input scorer names to counts of tracks omitted entirely
    (not tracks combined through a maximum or projected onto components).
    ``n_components_withheld`` counts requested PCA components not returned per
    scorer, including limits from track count, sample count and numerical rank.
    """

    def __new__(cls, items, n_dropped, n_components_withheld=None):
        result = super().__new__(cls, items)
        result.n_dropped = n_dropped
        result.n_components_withheld = n_components_withheld or {}
        return result

    def __reduce__(self):
        return (type(self), (tuple(self), self.n_dropped, self.n_components_withheld))


def _checked(values, track_scorer):
    values = np.asarray(values, dtype=np.float64)
    track_scorer = np.asarray(track_scorer)
    if values.ndim != 2 or track_scorer.shape != (values.shape[1],):
        raise ValueError("need a (variants, tracks) matrix and one scorer name per track")
    if not np.isfinite(values).all():
        raise MeasurementUnavailable("track values are not all finite")
    return values, track_scorer


def modality_max(values, track_scorer, modalities=AVI_MODALITIES):
    """Largest absolute score across each modality's tracks.

    Returns ``(features (N, M), names)`` with ``n_dropped`` counts per scorer.
    A modality is included only when all its scorers are present; a modality
    made of several scorers sums their maxima. Raises ValueError for mismatched
    shapes or no complete modality, MeasurementUnavailable for nonfinite scores.
    """
    values, track_scorer = _checked(values, track_scorer)
    present = set(track_scorer.tolist())
    columns, names, used = [], [], set()
    for name, scorers in modalities.items():
        scorers = tuple(scorers)
        if not scorers or not set(scorers) <= present:
            continue
        columns.append(sum(np.abs(values[:, track_scorer == s]).max(axis=1) for s in scorers))
        names.append(name)
        used.update(scorers)
    if not columns:
        raise MeasurementUnavailable("no modality has all of its scorers among the tracks")
    features = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise MeasurementUnavailable("modality features are not finite")
    dropped = {s: int((track_scorer == s).sum()) if s not in used else 0
               for s in dict.fromkeys(track_scorer.tolist())}
    return FeatureReduction((features, names), dropped)


def category_pca(query, reference, track_scorer, n_components=30):
    """Principal components per scorer, standardised and fitted on the reference.

    Returns ``(query_features, reference_features, names)`` with ``n_dropped``
    counts of constant tracks and ``n_components_withheld`` per scorer. Each
    scorer contributes at most its standardised reference numerical rank,
    capped by ``n_components`` and ``reference rows - 1`` components, named
    ``DNASE_pc1``, etc. Entirely constant categories contribute nothing.

    Raises ValueError for incompatible shapes or invalid component counts,
    MeasurementUnavailable for nonfinite values or an undefined reduction.
    """
    from manylatents.metrics import standardize_against

    query, track_scorer = _checked(query, track_scorer)
    reference, _ = _checked(reference, track_scorer)
    if not isinstance(n_components, Integral) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if reference.shape[0] < 2:
        raise MeasurementUnavailable("PCA needs at least two reference rows")
    query_parts, reference_parts, names, dropped = [], [], [], {}
    withheld = {}
    for scorer in dict.fromkeys(track_scorer.tolist()):
        columns = track_scorer == scorer
        varying = np.any(reference[:, columns] != reference[:1, columns], axis=0)
        dropped[scorer] = int((~varying).sum())
        if not varying.any():
            withheld[scorer] = int(n_components)
            continue
        q, r, _ = standardize_against(
            query[:, columns][:, varying], reference[:, columns][:, varying], constant="drop"
        )
        _, singular_values, components = np.linalg.svd(r, full_matrices=False)
        rank = int((singular_values > singular_values[0] * max(r.shape)
                    * np.finfo(r.dtype).eps).sum())
        k = min(n_components, rank, r.shape[0] - 1)
        withheld[scorer] = int(n_components - k)
        projected_q = q @ components[:k].T
        projected_r = r @ components[:k].T
        if not (np.isfinite(projected_q).all() and np.isfinite(projected_r).all()):
            raise MeasurementUnavailable("PCA projections are not finite")
        query_parts.append(projected_q)
        reference_parts.append(projected_r)
        names += [f"{scorer}_pc{i + 1}" for i in range(k)]
    if not names:
        raise MeasurementUnavailable("the reference does not vary in any track")
    return FeatureReduction((np.concatenate(query_parts, axis=1),
                             np.concatenate(reference_parts, axis=1), names), dropped, withheld)
