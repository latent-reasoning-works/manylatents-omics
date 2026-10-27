"""Objectives: the diffusion-operator match, the Jacobian penalty, heads and their sum."""
import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from manylatents.dogma.alphagenome.interventions import LowRankEdit  # noqa: E402
from manylatents.dogma.alphagenome.objectives import (  # noqa: E402
    Composite, CovarianceIsotropy, DiffusionOperatorMatch, JacobianPenalty, TrackHead,
    as_composite, diffusion_operator, jacobian_frobenius,
)


def _random(shape, seed=0):
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape), jnp.float32)


def _rotation(d, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return jnp.asarray(q, jnp.float32)


def test_diffusion_operator_is_row_stochastic():
    p = diffusion_operator(_random((12, 5)))
    np.testing.assert_allclose(np.asarray(p.sum(axis=1)), np.ones(12), rtol=1e-6)
    assert (np.asarray(p) > 0).all()


def test_diffusion_operator_match_properties():
    match = DiffusionOperatorMatch()
    s, t = _random((16, 6)), _random((16, 9), seed=1)
    assert float(match.loss(s, s)) == pytest.approx(0.0, abs=1e-6)
    value = float(match.loss(s, t))
    assert value > 1e-3
    assert float(match.loss(t, s)) == pytest.approx(value, rel=1e-5)
    # rotation and uniform scaling of the student leave the operator unchanged
    assert float(match.loss(s @ _rotation(6) * 7.5, t)) == pytest.approx(value, rel=1e-4)
    assert float(match.loss(s @ _rotation(6, 2), s)) == pytest.approx(0.0, abs=1e-5)


def test_diffusion_operator_match_has_finite_gradients():
    match = DiffusionOperatorMatch()
    s, t = _random((8, 4)), _random((8, 3), seed=1)
    grad = jax.grad(lambda x: match.loss(x, t))(s)
    assert np.isfinite(np.asarray(grad)).all() and np.abs(np.asarray(grad)).sum() > 0
    # with S == T the loss is at its minimum: the gradient vanishes
    zero = jax.grad(lambda x: match.loss(x, s))(s)
    np.testing.assert_allclose(np.asarray(zero), 0, atol=1e-5)


def test_diffusion_operator_match_reads_outputs():
    match = DiffusionOperatorMatch(layers=["a", "b"], source="delta")
    a, b = _random((10, 3)), _random((10, 2), seed=2)
    outputs = {"delta": {"a": a, "b": b}}
    assert float(match(outputs, {"teacher": jnp.concatenate([a, b], 1)})) == pytest.approx(
        0.0, abs=1e-6)


def test_jacobian_of_a_linear_map():
    a = _random((5, 7))

    def fn(h):
        return a @ h

    x = _random((4, 7), seed=1)
    expected = float(jnp.sum(a ** 2))
    assert float(jacobian_frobenius(fn, x, mode="exact")) == pytest.approx(expected, rel=1e-5)
    estimate = float(jacobian_frobenius(fn, x, jax.random.PRNGKey(0), probes=2000))
    assert estimate == pytest.approx(expected, rel=0.05)
    with pytest.raises(ValueError, match="rng"):
        jacobian_frobenius(fn, x)


def test_jacobian_hutchinson_matches_exact_on_a_nonlinear_map():
    w = _random((6, 6), seed=3)

    def fn(h):
        return jnp.tanh(w @ h)

    x = _random((3, 6), seed=4)
    exact = float(jacobian_frobenius(fn, x, mode="exact"))
    estimate = float(jacobian_frobenius(fn, x, jax.random.PRNGKey(1), probes=4000))
    assert estimate == pytest.approx(exact, rel=0.05)


def test_jacobian_penalty_on_the_edit():
    edit = LowRankEdit(rank=2)
    params = edit.init(jax.random.PRNGKey(0), 8)
    x = _random((4, 8))
    outputs = {"edit_fn": lambda h: edit.apply(params, h), "edit_input": x,
               "rng": jax.random.PRNGKey(2)}
    # at init the edit is the identity: ||I||_F^2 = d, and 0 for the residual
    assert float(JacobianPenalty(mode="exact")(outputs, {})) == pytest.approx(8.0, rel=1e-5)
    assert float(JacobianPenalty(mode="exact", residual=True)(outputs, {})) == pytest.approx(
        0.0, abs=1e-6)
    assert float(JacobianPenalty(probes=4)(outputs, {})) == pytest.approx(8.0, rel=1e-5)


@pytest.mark.parametrize("loss", ["mse", "poisson", "cosine"])
@pytest.mark.parametrize("hidden", [0, 5])
def test_track_head(loss, hidden):
    head = TrackHead(layers=["a", "b"], source="delta", loss=loss, hidden=hidden)
    outputs = {"delta": {"a": _random((6, 3)), "b": _random((6, 4), seed=1)}}
    target = jnp.abs(_random((6, 2), seed=2))
    batch = {"target": target}
    params = head.init(jax.random.PRNGKey(0), outputs, batch)
    assert head.predict(params, outputs).shape == (6, 2)
    value = head(outputs, batch, params)
    assert np.isfinite(float(value))
    if loss == "poisson":
        assert (np.asarray(head.predict(params, outputs)) > 0).all()


def test_track_head_fits_a_linear_target():
    import optax

    x = _random((32, 4))
    target = x @ _random((4, 3), seed=1)
    head = TrackHead(layers="a", source="ref")
    outputs, batch = {"ref": {"a": x}}, {"target": target}
    params = head.init(jax.random.PRNGKey(0), outputs, batch)
    opt = optax.adam(0.05)
    state = opt.init(params)
    first = float(head(outputs, batch, params))
    for _ in range(300):
        grads = jax.grad(lambda p: head(outputs, batch, p))(params)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
    assert float(head(outputs, batch, params)) < 1e-3 * first


def test_covariance_isotropy():
    iso = CovarianceIsotropy(layers="a", source="ref")
    rng = np.random.default_rng(0)
    independent = jnp.asarray(rng.normal(size=(500, 4)) * [1, 2, 3, 4], jnp.float32)
    mixed = independent @ _random((4, 4), seed=5)
    low = float(iso({"ref": {"a": independent}}, {}))
    high = float(iso({"ref": {"a": mixed}}, {}))
    assert 0 <= low < 0.05 < high < 1


def test_composite_weights_and_names():
    a = TrackHead(layers="x", source="ref", weight=2.0, name="first")
    b = CovarianceIsotropy(layers="x", source="ref", weight=0.5)
    both = Composite([a, b])
    outputs = {"ref": {"x": _random((8, 3))}}
    batch = {"target": _random((8, 2), seed=1)}
    params = both.init(jax.random.PRNGKey(0), outputs, batch)
    assert set(params) == {"first", "covariance_isotropy"}
    terms = both.terms(outputs, batch, params)
    assert float(both(outputs, batch, params)) == pytest.approx(
        2.0 * float(terms["first"]) + 0.5 * float(terms["covariance_isotropy"]), rel=1e-6)
    assert both.batch_keys == ("target",)
    assert both.find(TrackHead) is a
    with pytest.raises(ValueError, match="distinct"):
        Composite([TrackHead(), TrackHead()])
    assert as_composite(a).objectives == [a] and as_composite(both) is both


def test_bad_options():
    with pytest.raises(ValueError, match="source"):
        TrackHead(source="both")
    with pytest.raises(ValueError, match="loss"):
        TrackHead(loss="huber")
    with pytest.raises(ValueError, match="mode"):
        JacobianPenalty(mode="approx")
