"""Per-variant per-layer signal record schema for the dogma signal manifold.

This is the typed data contract that shapes the DNA arm (AlphaGenome regulatory
tracks) like the protein arm, so ONE geometry engine serves both. The spine is:

    DNA sequence -> AlphaGenome tracks -> SignalRecord(variant, track)
                 -> per-variant per-layer vector -> manifold geometry

A ``SignalRecord`` is emitted once per (variant, track). Records are then reduced
per (variant, layer) into a fixed-width vector. The reducer output is a
``Dict[layer -> (N_variants, LAYER_VECTOR_DIM)]`` channel dict whose shape matches
exactly what the fusion algorithms already consume:

  * ``manylatents.dogma.algorithms.learned_fusion.AutoencoderFusion`` takes an
    ``embeddings`` dict keyed by *channel* -> ``(N, dim)`` array/tensor and
    concatenates channels in dict order (see its ``_prepare_data``). Here each
    LAYER is a channel, so ``stack_layer_matrix`` output feeds it directly.
  * ``CentralDogmaFusion`` concatenates named modality embeddings in a fixed
    order; ``reduce_variant_vector`` produces the same flat concatenation over
    layers in ``CANONICAL_LAYERS`` order.

Design note (deliberately minimal — "less abstraction over more"): everything is
numpy + stdlib. No torch, no AlphaGenome, no GPU imports, so this module and its
smoke test run under a plain ``python``/``pytest``. Fusion accepts numpy arrays
(it converts np -> torch internally), so returning numpy keeps the seam clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Layer taxonomy
# ---------------------------------------------------------------------------

# Canonical layers, in the FIXED order used to lay out the per-variant vector.
# The `rna` slot is first-class (an Orthrus RNA encoder + sequences_rna.yaml
# already exist) and is always present even when a variant has no RNA tracks.
CANONICAL_LAYERS: Tuple[str, ...] = (
    "accessibility",  # open-chromatin (ATAC / DNase)
    "tf",             # transcription-factor binding (ChIP-TF)
    "histone",        # histone marks (ChIP-histone)
    "cage",           # promoter / TSS activity (CAGE)
    "rna",            # expression (RNA-seq)  -- kept first-class
    "splice",         # splice-site usage
)

# AlphaGenome output-type -> layer. Keys are lowercased AlphaGenome OutputType
# names as `AlphaGenomeEncoder.predict()` returns them (it keys results by
# `ot.name.lower()`), plus a few obvious aliases. Normalized via
# `output_type_to_layer` so callers may pass "ATAC", "atac", "RNA_SEQ", etc.
LAYER_TAXONOMY: Dict[str, str] = {
    # accessibility
    "atac": "accessibility",
    "dnase": "accessibility",
    # transcription factors
    "chip_tf": "tf",
    "tf": "tf",
    # histone marks
    "chip_histone": "histone",
    "histone": "histone",
    # CAGE / TSS
    "cage": "cage",
    # RNA expression / 3' processing
    "rna_seq": "rna",
    "rna": "rna",
    "polya": "rna",  # polyadenylation (RNA 3' processing) -> rna slot
    # splicing
    "splice_sites": "splice",
    "splice_site_usage": "splice",
    "splice_junctions": "splice",
    "splice_donor": "splice",     # in #19's coding-grid TRACK_SUBSET
    "splice_acceptor": "splice",  # in #19's coding-grid TRACK_SUBSET
    "splice": "splice",
}

# Small constant so the log2 fold-change is defined for tracks that are ~0 away
# from active regions (matches #19's ``compute_track_delta`` eps).
_EPS: float = 1e-6


class UnknownOutputTypeError(KeyError):
    """Raised when an AlphaGenome output type has no mapped layer."""


def output_type_to_layer(output_type: str) -> str:
    """Map an AlphaGenome output-type name to a canonical layer.

    Case-insensitive; accepts enum-style names (``"RNA_SEQ"``) and the
    lowercased track keys that ``AlphaGenomeEncoder.predict()`` returns
    (``"rna_seq"``).

    Raises:
        UnknownOutputTypeError: if the output type is not in ``LAYER_TAXONOMY``.
            Explicit failure is intentional (defer a silent fallback) so a new
            AlphaGenome head is a deliberate one-line addition here.
    """
    key = output_type.strip().lower()
    try:
        return LAYER_TAXONOMY[key]
    except KeyError as exc:
        raise UnknownOutputTypeError(
            f"No layer mapped for AlphaGenome output type {output_type!r}. "
            f"Known: {sorted(set(LAYER_TAXONOMY))}"
        ) from exc


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantKey:
    """Identity of a variant. ``id`` is the stable join key across arms."""

    chrom: str
    pos: int
    ref: str
    alt: str
    id: str

    def as_tuple(self) -> Tuple[str, int, str, str, str]:
        return (self.chrom, self.pos, self.ref, self.alt, self.id)


# Public alias: the oracle contract (#29) refers to this type as ``Variant``.
Variant = VariantKey


@dataclass(frozen=True)
class SignalRecord:
    """One (variant, track) signal: the atomic unit of the signal manifold.

    Attributes:
        variant: variant identity (chrom, pos, ref, alt, id).
        track_id: AlphaGenome track identifier (e.g. ``"ATAC:K562"``).
        layer: canonical layer, one of ``CANONICAL_LAYERS``.
        cell_type: biosample / ontology term for the track, if known.
        delta: scalar track effect, ref->alt (signed; positive = alt increases).
        effect_pctl: percentile [0,100] of |delta| vs the scoring population.
        activity_pctl: percentile [0,100] of the track's ref (WT) activity.
    """

    variant: VariantKey
    track_id: str
    layer: str
    cell_type: Optional[str]
    delta: float
    effect_pctl: float
    activity_pctl: float

    def __post_init__(self) -> None:
        if self.layer not in CANONICAL_LAYERS:
            raise ValueError(
                f"layer must be one of {CANONICAL_LAYERS}, got {self.layer!r}"
            )


# ---------------------------------------------------------------------------
# Builder: AlphaGenome-like prediction pair -> SignalRecords
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackMeta:
    """Optional per-output-type track metadata, aligned to the track axis.

    ``cell_types`` and ``track_ids`` are length ``n_tracks``; either may be None
    to fall back to synthesized ids/None cell types.
    """

    cell_types: Optional[Sequence[Optional[str]]] = None
    track_ids: Optional[Sequence[str]] = None


def _collapse_positions(arr: np.ndarray, how: str) -> np.ndarray:
    """Collapse a (positions, n_tracks) array over positions -> (n_tracks,)."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:  # already (n_tracks,) — a pre-reduced scalar per track
        return a
    if a.ndim != 2:
        raise ValueError(f"track array must be 1D or 2D, got shape {a.shape}")
    if how == "mean":
        return a.mean(axis=0)
    if how == "sum":
        return a.sum(axis=0)
    if how == "max":
        return a.max(axis=0)
    raise ValueError(f"how must be 'mean', 'sum', or 'max', got {how!r}")


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,100] of each value *within* ``values``.

    WITHIN-CALL fallback used only when no background is supplied. It ranks a
    variant's tracks against each other, which couples layers: a big effect in
    one layer depresses the ranks of the others (empirically leaks class signal
    across layers). Prefer ``_percentile_against`` with a real background panel.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n <= 1:
        return np.full(n, 50.0)
    order = v.argsort(kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return ranks / (n - 1) * 100.0


def _percentile_against(values: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Percentile in [0,100] of each value against a fixed ``background`` sample.

    ``pctl = 100 * (# background <= value) / len(background)``. Each track is
    ranked independently of the variant's other tracks, so layers no longer leak
    signal into one another. This is the "roll with ours" background panel —
    e.g. |log2FC| over a random-variant reference set for ``effect``, and WT
    activity over a genome sample for ``activity``.
    """
    bg = np.sort(np.asarray(background, dtype=float))
    v = np.asarray(values, dtype=float)
    if bg.size == 0:
        return np.full(v.shape, 50.0)
    idx = np.searchsorted(bg, v, side="right")
    return idx / bg.size * 100.0


def build_signal_records(
    variant: VariantKey,
    ref_pred: Mapping[str, np.ndarray],
    alt_pred: Mapping[str, np.ndarray],
    track_meta: Optional[Mapping[str, TrackMeta]] = None,
    reduce: str = "mean",
    effect_background: Optional[np.ndarray] = None,
    activity_background: Optional[np.ndarray] = None,
) -> List[SignalRecord]:
    """Build SignalRecords for one variant from an AlphaGenome-like pred pair.

    ``ref_pred`` / ``alt_pred`` mirror ``AlphaGenomeEncoder.predict()`` output:
    a dict keyed by output type (e.g. ``"atac"``, ``"rna_seq"``) mapping to a
    ``(positions, n_tracks)`` array (or a pre-reduced ``(n_tracks,)`` array).

    ``delta`` is the signed **log2 fold-change** of the position-collapsed track,
    ``log2((alt+eps)/(ref+eps))`` — the AlphaGenome/chorus standard and the same
    per-track quantity #19's coding-grid scalar maxes over (see
    :func:`variant_scalar_delta`). Track values are clamped to >= 0 first
    (AlphaGenome signal is non-negative).

    ``effect_pctl`` / ``activity_pctl`` rank against a fixed ``*_background``
    sample when supplied (tracks stay independent); otherwise they fall back to
    a within-call rank (a placeholder that leaks signal across layers).

    Args:
        variant: variant identity.
        ref_pred: reference-sequence predictions per output type.
        alt_pred: alternate-sequence predictions per output type (same keys/shapes).
        track_meta: optional per-output-type cell types / track ids.
        reduce: how to collapse the position axis into a per-track scalar.
        effect_background: 1-D sample of |log2FC| to rank effect percentiles
            against (e.g. from a random-variant panel). None -> within-call rank.
        activity_background: 1-D sample of WT activity to rank activity
            percentiles against. None -> within-call rank.

    Returns:
        List of SignalRecords, one per (output_type, track).
    """
    track_meta = track_meta or {}

    # First pass: per (output_type, track) collect delta + ref activity so the
    # percentiles can be ranked across the whole variant's track population.
    raw: List[dict] = []
    deltas: List[float] = []
    activities: List[float] = []

    for output_type, ref_arr in ref_pred.items():
        if output_type not in alt_pred:
            raise KeyError(
                f"output type {output_type!r} present in ref_pred but not alt_pred"
            )
        layer = output_type_to_layer(output_type)
        # AlphaGenome signal is non-negative; clamp so log2FC is well-defined.
        ref_scalar = np.maximum(_collapse_positions(ref_arr, reduce), 0.0)
        alt_scalar = np.maximum(_collapse_positions(alt_pred[output_type], reduce), 0.0)
        if ref_scalar.shape != alt_scalar.shape:
            raise ValueError(
                f"ref/alt track count mismatch for {output_type!r}: "
                f"{ref_scalar.shape} vs {alt_scalar.shape}"
            )

        meta = track_meta.get(output_type, TrackMeta())
        n_tracks = ref_scalar.shape[0]
        for t in range(n_tracks):
            # Signed log2 fold-change, ref->alt (matches #19's compute_track_delta).
            delta = float(np.log2((alt_scalar[t] + _EPS) / (ref_scalar[t] + _EPS)))
            activity = float(ref_scalar[t])
            cell_type = (
                meta.cell_types[t]
                if meta.cell_types is not None and t < len(meta.cell_types)
                else None
            )
            track_id = (
                meta.track_ids[t]
                if meta.track_ids is not None and t < len(meta.track_ids)
                else f"{output_type}:{t}"
            )
            raw.append(
                {
                    "track_id": track_id,
                    "layer": layer,
                    "cell_type": cell_type,
                    "delta": delta,
                    "activity": activity,
                }
            )
            deltas.append(delta)
            activities.append(activity)

    abs_deltas = np.abs(np.asarray(deltas))
    if effect_background is not None:
        effect_pctls = _percentile_against(abs_deltas, effect_background)
    else:
        effect_pctls = _percentile_rank(abs_deltas)
    if activity_background is not None:
        activity_pctls = _percentile_against(np.asarray(activities), activity_background)
    else:
        activity_pctls = _percentile_rank(np.asarray(activities))

    return [
        SignalRecord(
            variant=variant,
            track_id=r["track_id"],
            layer=r["layer"],
            cell_type=r["cell_type"],
            delta=r["delta"],
            effect_pctl=float(effect_pctls[i]),
            activity_pctl=float(activity_pctls[i]),
        )
        for i, r in enumerate(raw)
    ]


# ---------------------------------------------------------------------------
# Reducer: (variant, layer) -> fixed vector for geometry
# ---------------------------------------------------------------------------

# Fixed per-layer aggregation. Signed mean keeps direction; max-abs keeps the
# strongest track effect; pctl means summarize rank. Fixed width => same shape
# for every variant, so the geometry engine sees a regular grid.
LAYER_STATS: Tuple[str, ...] = (
    "delta_mean",
    "delta_maxabs",
    "effect_pctl_mean",
    "activity_pctl_mean",
)
LAYER_VECTOR_DIM: int = len(LAYER_STATS)


def reduce_layer_vector(records: Sequence[SignalRecord]) -> np.ndarray:
    """Aggregate records of a SINGLE (variant, layer) into a fixed vector.

    Empty input -> zeros (so an absent layer still occupies its fixed slot).
    """
    vec = np.zeros(LAYER_VECTOR_DIM, dtype=float)
    if not records:
        return vec
    deltas = np.array([r.delta for r in records], dtype=float)
    effect = np.array([r.effect_pctl for r in records], dtype=float)
    activity = np.array([r.activity_pctl for r in records], dtype=float)
    vec[0] = deltas.mean()
    vec[1] = deltas[np.argmax(np.abs(deltas))]  # signed value of max-|delta| track
    vec[2] = effect.mean()
    vec[3] = activity.mean()
    return vec


def reduce_variant_layers(records: Sequence[SignalRecord]) -> Dict[str, np.ndarray]:
    """Per-variant reducer: {layer -> (LAYER_VECTOR_DIM,)} for ALL layers.

    Every canonical layer is present (zeros when the variant has no such track),
    so downstream shape is fixed and the ``rna`` slot is always first-class.
    """
    by_layer: Dict[str, List[SignalRecord]] = {ly: [] for ly in CANONICAL_LAYERS}
    for r in records:
        by_layer[r.layer].append(r)
    return {ly: reduce_layer_vector(by_layer[ly]) for ly in CANONICAL_LAYERS}


def reduce_variant_vector(records: Sequence[SignalRecord]) -> np.ndarray:
    """Flat per-variant vector: layer vectors concatenated in CANONICAL_LAYERS
    order. Shape ``(len(CANONICAL_LAYERS) * LAYER_VECTOR_DIM,)``.

    Mirrors ``CentralDogmaFusion``'s fixed-order concatenation, with layers in
    place of modalities.
    """
    per_layer = reduce_variant_layers(records)
    return np.concatenate([per_layer[ly] for ly in CANONICAL_LAYERS])


def stack_layer_matrix(
    records: Sequence[SignalRecord],
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Stack records across variants into a fusion-ready channel dict.

    Returns:
        (variant_ids, channels) where ``variant_ids`` is the row order and
        ``channels`` maps each canonical layer -> ``(N_variants, LAYER_VECTOR_DIM)``
        array. This is exactly the ``embeddings=`` dict shape consumed by
        ``AutoencoderFusion`` / ``FrobeniusAEFusion`` (layers as channels), and
        row ``i`` aligns across all layers for the same variant.
    """
    # Preserve first-seen variant order.
    variant_ids: List[str] = []
    grouped: Dict[str, List[SignalRecord]] = {}
    for r in records:
        vid = r.variant.id
        if vid not in grouped:
            grouped[vid] = []
            variant_ids.append(vid)
        grouped[vid].append(r)

    channels: Dict[str, np.ndarray] = {ly: [] for ly in CANONICAL_LAYERS}
    for vid in variant_ids:
        per_layer = reduce_variant_layers(grouped[vid])
        for ly in CANONICAL_LAYERS:
            channels[ly].append(per_layer[ly])

    stacked = {
        ly: (
            np.vstack(rows)
            if rows
            else np.zeros((0, LAYER_VECTOR_DIM), dtype=float)
        )
        for ly, rows in channels.items()
    }
    return variant_ids, stacked


# ---------------------------------------------------------------------------
# Scalar bridge: the single value #19's coding grid (Gate D4) scores per variant
# ---------------------------------------------------------------------------

# #19's TRACK_SUBSET = (cage, rna_seq, polya, splice_donor, splice_acceptor)
# maps, via LAYER_TAXONOMY, onto exactly these canonical layers. Keeping the
# scalar as a reduction of SignalRecords means the grid value equals a collapse
# of the per-layer manifold — one Δ definition (log2FC), one source of truth.
SCALAR_DELTA_LAYERS: Tuple[str, ...] = ("cage", "rna", "splice")


def variant_scalar_delta(
    records: Sequence[SignalRecord],
    layers: Sequence[str] = SCALAR_DELTA_LAYERS,
    reduce: str = "maxabs",
) -> float:
    """Collapse one variant's records to the scalar the coding grid (#19) scores.

    ``reduce="maxabs"`` reproduces #19's ``max(|log2FC|)`` over its TRACK_SUBSET
    (here selected by canonical layer). This is the seam that keeps the scalar
    grid entry and the per-layer manifold consistent — #19 imports this instead
    of its own ``compute_track_delta``.
    """
    keep = set(layers)
    vals = np.abs(np.array([r.delta for r in records if r.layer in keep], dtype=float))
    if vals.size == 0:
        return 0.0
    if reduce == "maxabs":
        return float(vals.max())
    if reduce == "meanabs":
        return float(vals.mean())
    raise ValueError(f"reduce must be 'maxabs' or 'meanabs', got {reduce!r}")


__all__ = [
    "CANONICAL_LAYERS",
    "LAYER_TAXONOMY",
    "LAYER_STATS",
    "LAYER_VECTOR_DIM",
    "SCALAR_DELTA_LAYERS",
    "UnknownOutputTypeError",
    "output_type_to_layer",
    "VariantKey",
    "Variant",
    "SignalRecord",
    "TrackMeta",
    "build_signal_records",
    "reduce_layer_vector",
    "reduce_variant_layers",
    "reduce_variant_vector",
    "stack_layer_matrix",
    "variant_scalar_delta",
]
