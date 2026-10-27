"""Scalar features of allele responses: delta norm, cosine and pr3."""
import numpy as np
import pytest

from manylatents.dogma.latent_features import cosine, delta_norm, participation_ratio, pr3


def test_delta_norm_and_cosine_keep_leading_axes():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 3, 7)), rng.normal(size=(5, 3, 7))
    assert delta_norm(a, b).shape == (5, 3)
    np.testing.assert_allclose(delta_norm(a, b), np.sqrt(((b - a) ** 2).sum(-1)))
    expected = (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    np.testing.assert_allclose(cosine(a, b), expected)


def test_cosine_nan_when_a_norm_is_zero():
    out = cosine(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 2.0], [0.0, 3.0]]))
    assert np.isnan(out[0]) and out[1] == 0.0
    assert np.isnan(cosine(np.ones(3), np.zeros(3)))


def test_float16_input_computed_in_float64():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    assert delta_norm(a, a + 1).dtype == np.float64


@pytest.mark.parametrize("seed", range(5))
def test_pr3_closed_form(seed):
    rng = np.random.default_rng(seed)
    ref, alt, alt2, alt3 = rng.normal(size=(4, 9))
    rows = np.stack([alt - ref, alt2 - ref, alt3 - ref])
    centred = rows - rows.mean(0)
    lam = np.linalg.svd(centred, compute_uv=False) ** 2
    r = lam[1] / lam[0]
    value = pr3(ref, alt, alt2, alt3)
    np.testing.assert_allclose(value, (1 + r) ** 2 / (1 + r ** 2))
    np.testing.assert_allclose(value, lam.sum() ** 2 / (lam ** 2).sum())
    assert 1.0 <= value <= 2.0


def test_pr3_extremes_and_degenerate():
    ref = np.zeros(2)
    # collinear alternates: one direction
    assert pr3(ref, np.array([1.0, 0]), np.array([2.0, 0]), np.array([4.0, 0])) == pytest.approx(1.0)
    # equilateral triangle: two equal directions
    angles = np.deg2rad([90, 210, 330])
    alt, alt2, alt3 = np.stack([np.cos(angles), np.sin(angles)], axis=1) + 5.0
    assert pr3(ref, alt, alt2, alt3) == pytest.approx(2.0)
    # all alternates equal: undefined
    same = np.array([1.0, 1.0])
    assert np.isnan(pr3(ref, same, same, same))


def test_pr3_does_not_depend_on_ref_and_batches():
    rng = np.random.default_rng(1)
    h = rng.normal(size=(4, 6, 11))
    batched = pr3(*h)
    assert batched.shape == (6,)
    np.testing.assert_allclose(batched, pr3(h[0] + 3.0, *h[1:]))
    np.testing.assert_allclose(batched[2], pr3(*h[:, 2]))
    np.testing.assert_allclose(participation_ratio(np.stack(list(h[1:]), axis=-2) - h[0][..., None, :]),
                               batched)
