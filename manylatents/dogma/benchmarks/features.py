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


def _reduction(columns, names, dropped):
    if not columns:
        raise MeasurementUnavailable("no usable tracks for this reduction")
    values = np.column_stack(columns)
    if not np.isfinite(values).all():
        raise MeasurementUnavailable("reduced features are not finite")
    return FeatureReduction((values, names), dropped)


def signed_extremes(values, track_scorer, signed_scorers):
    """Positive maximum and negative minimum per signed scorer, zero if absent.

    Scorers follow input order. Unsigned tracks are omitted and counted in
    ``n_dropped``. Returns ``(features, names)`` as a FeatureReduction.
    """
    values, track_scorer = _checked(values, track_scorer)
    signed_scorers = set(signed_scorers)
    columns, names, dropped = [], [], {}
    for scorer in dict.fromkeys(track_scorer.tolist()):
        selected = track_scorer == scorer
        dropped[scorer] = 0 if scorer in signed_scorers else int(selected.sum())
        if scorer in signed_scorers:
            block = values[:, selected]
            columns.extend([np.maximum(block.max(axis=1), 0), np.minimum(block.min(axis=1), 0)])
            names.extend([f"{scorer}_positive", f"{scorer}_negative"])
    return _reduction(columns, names, dropped)


def _reference_blocks(query, reference, track_scorer):
    query, track_scorer = _checked(query, track_scorer)
    reference, _ = _checked(reference, track_scorer)
    if len(reference) < 2:
        raise MeasurementUnavailable("standardisation needs at least two reference rows")
    blocks, dropped = [], {}
    for scorer in dict.fromkeys(track_scorer.tolist()):
        selected = track_scorer == scorer
        q, r = query[:, selected], reference[:, selected]
        varying = np.any(r != r[:1], axis=0)
        dropped[scorer] = int((~varying).sum())
        if not varying.any():
            continue
        q, r = q[:, varying], r[:, varying]
        mean, spread = r.mean(axis=0), r.std(axis=0)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            z = (q - mean) / spread
        if not (np.isfinite(mean).all() and np.isfinite(spread).all()
                and (spread > 0).all() and np.isfinite(z).all()):
            raise MeasurementUnavailable("reference standardisation is not finite")
        blocks.append((scorer, q, r, z))
    return blocks, dropped


def modality_breadth(query, reference, track_scorer, quantile=0.99):
    """Fraction above reference absolute quantiles and effective track fraction.

    Each input scorer supplies two features. Reference-constant tracks are
    dropped and counted for both features. Standardisation uses population
    standard deviation. Effective fraction is (sum(z²))² / sum(z⁴) / track
    count; an all-zero row gives zero. Returns a FeatureReduction.
    """
    if not np.isscalar(quantile) or not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be finite and between zero and one")
    blocks, dropped = _reference_blocks(query, reference, track_scorer)
    columns, names = [], []
    for scorer, q, r, z in blocks:
        above = (np.abs(q) > np.quantile(np.abs(r), quantile, axis=0)).mean(axis=1)
        # Row scaling cancels in the ratio and prevents overflowing z**4.
        scale = np.abs(z).max(axis=1, keepdims=True)
        bounded = np.divide(z, scale, out=np.zeros_like(z), where=scale != 0)
        square = bounded ** 2
        numerator, denominator = square.sum(axis=1) ** 2, (square ** 2).sum(axis=1)
        effective = np.divide(numerator, denominator, out=np.zeros(len(q)), where=denominator != 0)
        columns.extend([above, effective / z.shape[1]])
        names.extend([f"{scorer}_fraction_above_background", f"{scorer}_effective_fraction"])
    return _reduction(columns, names, dropped)


def normalised_maxima(query, reference, track_scorer):
    """Maximum absolute reference-standardised score per input scorer.

    Reference-constant tracks are dropped and counted. Scorers with no retained
    track supply no feature; an entirely constant reference is unavailable.
    """
    blocks, dropped = _reference_blocks(query, reference, track_scorer)
    return _reduction([np.abs(z).max(axis=1) for _, _, _, z in blocks],
                      [f"{s}_normalised_max" for s, _, _, _ in blocks], dropped)


def group_maxima(values, track_scorer, track_group):
    """Maximum absolute score for each present (scorer, group) combination.

    Scorers and groups follow first occurrence. Names are ``scorer::group``.
    Missing group labels are unavailable; callers must assign their explicit
    catch-all group before reduction. No track is omitted.
    """
    values, track_scorer = _checked(values, track_scorer)
    track_group = np.asarray(track_group, dtype=object)
    if track_group.shape != track_scorer.shape:
        raise ValueError("need one group label per track")
    if any(not isinstance(g, str) or not g.strip() for g in track_group):
        raise MeasurementUnavailable("track group labels must be nonempty strings")
    columns, names = [], []
    for scorer in dict.fromkeys(track_scorer.tolist()):
        selected = track_scorer == scorer
        for group in dict.fromkeys(track_group[selected].tolist()):
            columns.append(np.abs(values[:, selected & (track_group == group)]).max(axis=1))
            names.append(f"{scorer}::{group}")
    return _reduction(columns, names, {s: 0 for s in dict.fromkeys(track_scorer.tolist())})
