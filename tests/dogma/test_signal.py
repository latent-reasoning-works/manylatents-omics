"""Smoke tests for the per-variant per-layer SignalRecord schema.

Fully synthetic: builds SignalRecords from an AlphaGenome-*like* prediction dict
and produces the per-variant per-layer vector. No real AlphaGenome / GPU / torch
required -- numpy + pytest only.
"""
import numpy as np
import pytest

from manylatents.dogma.signal import (
    CANONICAL_LAYERS,
    LAYER_VECTOR_DIM,
    SignalRecord,
    TrackMeta,
    UnknownOutputTypeError,
    VariantKey,
    build_signal_records,
    output_type_to_layer,
    reduce_variant_layers,
    reduce_variant_vector,
    stack_layer_matrix,
)


def _synthetic_ag_output(seed: int, n_pos: int = 8):
    """AlphaGenome-like output pair for one variant.

    Mirrors ``AlphaGenomeEncoder.predict()``: dict keyed by lowercased output
    type -> (positions, n_tracks) array. We return a (ref, alt) pair plus
    per-output-type track metadata.
    """
    rng = np.random.default_rng(seed)
    output_types = {
        "atac": 2,        # -> accessibility
        "dnase": 1,       # -> accessibility (second source, same layer)
        "chip_tf": 3,     # -> tf
        "chip_histone": 2,  # -> histone
        "cage": 1,        # -> cage
        "rna_seq": 2,     # -> rna
        "splice_sites": 1,  # -> splice
    }
    ref, alt, meta = {}, {}, {}
    for ot, n_tracks in output_types.items():
        base = rng.random((n_pos, n_tracks))
        ref[ot] = base
        # alt = ref plus a small perturbation so deltas are nonzero
        alt[ot] = base + rng.normal(0, 0.1, size=base.shape)
        meta[ot] = TrackMeta(
            cell_types=[f"cell{t}" for t in range(n_tracks)],
            track_ids=[f"{ot.upper()}:cell{t}" for t in range(n_tracks)],
        )
    return ref, alt, meta


def test_taxonomy_covers_all_alphagenome_heads():
    """Every AlphaGenome head maps onto a canonical layer."""
    assert output_type_to_layer("ATAC") == "accessibility"
    assert output_type_to_layer("dnase") == "accessibility"
    assert output_type_to_layer("CHIP_TF") == "tf"
    assert output_type_to_layer("chip_histone") == "histone"
    assert output_type_to_layer("CAGE") == "cage"
    assert output_type_to_layer("RNA_SEQ") == "rna"
    assert output_type_to_layer("splice_sites") == "splice"


def test_unknown_output_type_raises():
    """Unknown heads fail loudly (no silent fallback)."""
    with pytest.raises(UnknownOutputTypeError):
        output_type_to_layer("contact_maps")


def test_build_signal_records_shapes_and_layers():
    variant = VariantKey(chrom="chr1", pos=1000, ref="A", alt="T", id="v1")
    ref, alt, meta = _synthetic_ag_output(seed=0)

    records = build_signal_records(variant, ref, alt, track_meta=meta)

    # One record per track across all output types: 2+1+3+2+1+2+1 = 12
    assert len(records) == 12
    assert all(isinstance(r, SignalRecord) for r in records)
    assert all(r.layer in CANONICAL_LAYERS for r in records)
    # accessibility gets both ATAC (2) and DNase (1) => 3 records
    assert sum(r.layer == "accessibility" for r in records) == 3
    # percentiles are in [0, 100]
    assert all(0.0 <= r.effect_pctl <= 100.0 for r in records)
    assert all(0.0 <= r.activity_pctl <= 100.0 for r in records)
    # deltas are nonzero (alt perturbed)
    assert any(abs(r.delta) > 0 for r in records)


def test_reduce_variant_layers_fixed_shape_with_rna_slot():
    variant = VariantKey(chrom="chr1", pos=1000, ref="A", alt="T", id="v1")
    ref, alt, meta = _synthetic_ag_output(seed=1)
    records = build_signal_records(variant, ref, alt, track_meta=meta)

    per_layer = reduce_variant_layers(records)
    # All canonical layers present, each a fixed-width vector.
    assert set(per_layer) == set(CANONICAL_LAYERS)
    assert all(v.shape == (LAYER_VECTOR_DIM,) for v in per_layer.values())
    # rna slot is first-class and populated (rna_seq had 2 tracks).
    assert np.any(per_layer["rna"] != 0)

    flat = reduce_variant_vector(records)
    assert flat.shape == (len(CANONICAL_LAYERS) * LAYER_VECTOR_DIM,)


def test_empty_layer_slot_is_zeros():
    """A variant with only accessibility tracks still yields every slot."""
    variant = VariantKey(chrom="chr2", pos=50, ref="C", alt="G", id="only_atac")
    ref = {"atac": np.ones((4, 1))}
    alt = {"atac": np.ones((4, 1)) * 2}
    records = build_signal_records(variant, ref, alt)

    per_layer = reduce_variant_layers(records)
    assert np.any(per_layer["accessibility"] != 0)
    # untouched layers are exactly zero-filled slots
    for ly in ("tf", "histone", "cage", "rna", "splice"):
        assert np.array_equal(per_layer[ly], np.zeros(LAYER_VECTOR_DIM))


def test_stack_layer_matrix_is_fusion_ready():
    """Stacked channels match the AutoencoderFusion `embeddings` dict shape."""
    records = []
    for i, vid in enumerate(["v1", "v2", "v3"]):
        variant = VariantKey(chrom="chr1", pos=100 * i, ref="A", alt="T", id=vid)
        ref, alt, meta = _synthetic_ag_output(seed=i)
        records.extend(build_signal_records(variant, ref, alt, track_meta=meta))

    variant_ids, channels = stack_layer_matrix(records)

    assert variant_ids == ["v1", "v2", "v3"]
    assert set(channels) == set(CANONICAL_LAYERS)
    # each layer channel is (N_variants, LAYER_VECTOR_DIM)
    for ly, mat in channels.items():
        assert mat.shape == (3, LAYER_VECTOR_DIM), ly

    # Concatenating channels in canonical order reproduces per-variant vectors:
    # this is exactly what fusion does with `embeddings` (channels -> concat).
    concat = np.concatenate([channels[ly] for ly in CANONICAL_LAYERS], axis=1)
    assert concat.shape == (3, len(CANONICAL_LAYERS) * LAYER_VECTOR_DIM)


def test_stack_feeds_autoencoder_fusion_contract():
    """The channel dict is directly consumable as fusion `embeddings=`.

    We do not import torch here; instead we assert the exact contract that
    AutoencoderFusion._prepare_data relies on: a dict of channel -> 2D array
    with a common first (row) dimension. Kept torch-free so the smoke test runs
    anywhere.
    """
    records = []
    for i, vid in enumerate(["a", "b"]):
        variant = VariantKey(chrom="chrX", pos=i, ref="A", alt="C", id=vid)
        ref, alt, meta = _synthetic_ag_output(seed=10 + i)
        records.extend(build_signal_records(variant, ref, alt, track_meta=meta))

    _, channels = stack_layer_matrix(records)
    n_rows = {mat.shape[0] for mat in channels.values()}
    assert n_rows == {2}  # all channels share the row dimension
    assert all(isinstance(mat, np.ndarray) and mat.ndim == 2 for mat in channels.values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
