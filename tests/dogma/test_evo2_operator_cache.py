"""The teacher operator cache.

A teacher is frozen, encoded once, and stored as pooled representations plus
ONE bandwidth. Students then match the operator built from that cache. Three
properties are load-bearing and each has a test here: sigma is a single scalar
shared by both sides, the Gaussian diagonal survives, and a cache that does not
record how it was pooled is refused rather than silently consumed.
"""

import json

import pytest
import torch

from manylatents.dogma.algorithms.evo2_operator_cache import (
    load_teacher_cache,
    pair_and_window_sequences,
    soft_diffop,
    teacher_sigma,
    write_teacher_cache,
)
from manylatents.dogma.encoders.evo2 import Evo2Encoder


def _reference_operator(
    representations: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """An independent double-precision reference with no cdist and no matmul
    cancellation, so it can catch a regression `soft_diffop` itself would not."""
    reps = representations.double()
    squared_distances = ((reps.unsqueeze(1) - reps.unsqueeze(0)) ** 2).sum(-1)
    affinity = torch.exp(-squared_distances / (2 * sigma.double() ** 2))
    return (affinity / affinity.sum(-1, keepdim=True)).float()


def test_sigma_is_one_scalar():
    """The reference passes one teacher-derived sigma to teacher and student.
    A per-tensor sigma rescales with the student, so the objective cannot see a
    global rescaling at all."""
    sigma = teacher_sigma(torch.randn(64, 32), quantile=0.25)
    assert sigma.numel() == 1
    assert torch.isfinite(sigma) and sigma > 0


def test_sigma_is_blind_to_nothing_but_the_teacher():
    """Scaling the teacher scales its bandwidth: sigma is a length in the
    teacher's own space, not a normalised constant."""
    points = torch.randn(64, 32)
    assert torch.isclose(
        teacher_sigma(points * 10, quantile=0.25),
        teacher_sigma(points, quantile=0.25) * 10,
        rtol=1e-4,
    )


def test_operator_retains_its_diagonal_and_rows_sum_to_one():
    """Self-affinity is what makes K(x_i, x_j) = P_ij / P_ii valid. Zero the
    diagonal and every two-element subset operator becomes [[0, 1], [1, 0]]
    regardless of the data, so the m >= 2 hypothesis carries no information."""
    operator = soft_diffop(torch.randn(8, 4), sigma=torch.tensor(1.0))
    assert (operator.diagonal() > 0).all()
    assert torch.allclose(operator.sum(-1), torch.ones(8), atol=1e-5)


def test_operator_survives_a_translation_of_the_points():
    """The matmul cdist path cancels catastrophically above ~25 rows: a pure
    translation moved symmetric KL from 1.45e-11 to 0.697 in the text arm.
    Translating the inputs must leave the operator alone."""
    points = torch.randn(64, 8)
    sigma = teacher_sigma(points, quantile=0.25)
    near = soft_diffop(points, sigma)
    far = soft_diffop(points + 1000.0, sigma)
    assert torch.allclose(near, far, atol=1e-5), (near - far).abs().max()


def test_operator_is_computed_in_float32_from_bfloat16_inputs():
    """Pooled teacher activations arrive in bfloat16; the kernel must not be
    evaluated there."""
    operator = soft_diffop(
        torch.randn(16, 8, dtype=torch.bfloat16), sigma=torch.tensor(1.0)
    )
    assert operator.dtype == torch.float32


@pytest.mark.parametrize("embedding_dim", [1920, 4096, 8192])
def test_operator_matches_a_non_cdist_reference_at_production_width(embedding_dim):
    """1920/4096/8192 are the 1B/7B/40B embedding dims (Evo2Encoder.MODELS).
    The committed numerical tests above only exercise 4-32 columns; the matmul
    cdist path that cancels catastrophically above ~25 rows must also be
    checked at the widths teachers actually produce."""
    torch.manual_seed(0)
    representations = torch.randn(32, embedding_dim)
    sigma = teacher_sigma(representations, quantile=0.25)
    operator = soft_diffop(representations, sigma)
    reference = _reference_operator(representations, sigma)
    assert torch.allclose(operator, reference, atol=1e-5)


def test_bf16_pooled_representations_recover_the_float64_reference_operator():
    """A dtype check alone cannot catch a value regression: pool bfloat16
    hidden states through Evo2Encoder._pool_embeddings (production width,
    1920) the way a teacher run does, and check the resulting operator
    against an independent reference built from those same recovered values,
    not merely that the output happens to be labelled float32."""
    torch.manual_seed(0)
    hidden = torch.randn(4, 6, 1920, dtype=torch.bfloat16)
    encoder = Evo2Encoder(
        model_name="evo2_1b_base", layer_name="blocks.14.mlp.l3", device="cpu"
    )
    pooled = encoder._pool_embeddings(
        {"blocks.14.mlp.l3": hidden}, mask=torch.ones(4, 6, dtype=torch.bool)
    )
    sigma = teacher_sigma(pooled, quantile=0.25)
    operator = soft_diffop(pooled, sigma)
    reference = _reference_operator(pooled, sigma)
    assert torch.allclose(operator, reference, atol=1e-5)


def test_cache_without_a_pooling_marker_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"layer": "blocks.16.mlp.l3"}))
    torch.save(torch.randn(4, 4), tmp_path / "representations.pt")
    with pytest.raises(ValueError, match="pooling"):
        load_teacher_cache(tmp_path)


def test_cache_without_a_sigma_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"layer": "blocks.16.mlp.l3", "pooling": "masked_mean"})
    )
    torch.save(torch.randn(4, 4), tmp_path / "representations.pt")
    with pytest.raises(ValueError, match="sigma"):
        load_teacher_cache(tmp_path)


def test_round_trip_preserves_sigma_pooling_and_shape(tmp_path):
    reps = torch.randn(16, 8)
    write_teacher_cache(
        tmp_path,
        reps,
        sigma=torch.tensor(2.5),
        layer="blocks.16.mlp.l3",
        model_name="evo2_7b",
        window_bp=4096,
        pooling="masked_mean",
    )
    cache = load_teacher_cache(tmp_path)
    assert cache.pooling == "masked_mean"
    assert cache.layer == "blocks.16.mlp.l3"
    assert cache.model_name == "evo2_7b"
    assert cache.window_bp == 4096
    assert float(cache.sigma) == pytest.approx(2.5)
    assert cache.representations.shape == (16, 8)
    assert (tmp_path / "completed.json").is_file()


def test_write_refuses_a_non_scalar_sigma(tmp_path):
    with pytest.raises(ValueError, match="scalar"):
        write_teacher_cache(
            tmp_path,
            torch.randn(4, 4),
            sigma=torch.randn(4),
            layer="blocks.16.mlp.l3",
            model_name="evo2_7b",
            window_bp=4096,
            pooling="masked_mean",
        )


def test_pair_and_window_filters_ids_and_sequences_together():
    """A sequence missing from the FASTA parses as "" (ClinVarDataModule.
    get_sequences), the middle one here. Filtering IDs and sequences
    independently would keep IDs ["a", "c"] against a sequence list that
    dropped only the middle entry by value, so ID "c" would end up paired
    with sequence "CCCC" only by luck of position -- not by construction."""
    kept_ids, windowed, insufficient = pair_and_window_sequences(
        variant_ids=["a", "b", "c"],
        sequences=["AAAA", "", "CCCC"],
        window_bp=4,
    )
    assert kept_ids == ["a", "c"]
    assert windowed == ["AAAA", "CCCC"]
    assert insufficient == []


def test_pair_and_window_reports_insufficient_context_instead_of_silent_pass_through():
    """A sequence shorter than window_bp cannot supply that much context; an
    8192bp sweep measuring a 2048bp input would overstate the supported
    context, so short sequences are excluded and reported, not truncated in
    place and recorded as if they were full-length."""
    kept_ids, windowed, insufficient = pair_and_window_sequences(
        variant_ids=["a", "b"],
        sequences=["AAAA", "CC"],
        window_bp=4,
    )
    assert kept_ids == ["a"]
    assert windowed == ["AAAA"]
    assert insufficient == [("b", 2)]


def test_pair_and_window_centers_on_the_sequence_midpoint():
    seq = "GGG" + "A" * 4 + "TTT"  # variant sits in the middle 4bp window
    kept_ids, windowed, _ = pair_and_window_sequences(["v"], [seq], window_bp=4)
    assert kept_ids == ["v"]
    assert windowed == ["AAAA"]
