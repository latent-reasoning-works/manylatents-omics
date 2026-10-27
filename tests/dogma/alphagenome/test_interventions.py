"""Interventions inside the frozen forward of a tiny AlphaGenome-shaped model."""
import numpy as np
import pytest

pytest.importorskip("haiku")
pytest.importorskip("jmp")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from manylatents.dogma.alphagenome.interventions import (  # noqa: E402
    HeadOnly, LowRankEdit, make_forward,
)
from manylatents.dogma.alphagenome.taps import site_states  # noqa: E402
from tests.dogma.alphagenome.fake_model import LENGTH, make_model  # noqa: E402

F32 = "params=float32,compute=float32,output=float32"
SITE = LENGTH // 2 - 1


def _onehot(n, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.eye(4, dtype=np.float32)[rng.integers(0, 4, (n, LENGTH))])


def _close(actual, desired):
    """Equal up to float32 reassociation; the fake model's activations reach ~1e3."""
    actual, desired = np.asarray(actual), np.asarray(desired)
    np.testing.assert_allclose(actual, desired, rtol=1e-4,
                               atol=1e-5 * max(1.0, np.abs(desired).max()))


def _run(model, intervention, params, edit_layer, readouts, x, window=0, remat=False):
    forward = make_forward(model, intervention, edit_layer, readouts, SITE, window, remat)
    organism = jnp.zeros((x.shape[0],), jnp.int32)
    return jax.jit(forward)(params, model.params, model.state, x, organism)


def test_low_rank_edit_is_identity_at_init():
    edit = LowRankEdit(rank=3)
    params = edit.init(jax.random.PRNGKey(0), 16)
    h = jax.random.normal(jax.random.PRNGKey(1), (5, 7, 16))
    np.testing.assert_array_equal(np.asarray(edit.apply(params, h)), np.asarray(h))
    r = edit.rotation(params)
    assert r.shape == (3, 16)
    np.testing.assert_allclose(np.asarray(r @ r.T), np.eye(3), atol=1e-5)


def test_low_rank_edit_sets_the_subspace_and_stays_orthonormal():
    edit = LowRankEdit(rank=2)
    params = edit.init(jax.random.PRNGKey(0), 8)
    keys = jax.random.split(jax.random.PRNGKey(3), 3)
    params = {k: v + jax.random.normal(kk, v.shape) for (k, v), kk in zip(params.items(), keys)}
    r = edit.rotation(params)
    np.testing.assert_allclose(np.asarray(r @ r.T), np.eye(2), atol=1e-5)
    h = jax.random.normal(jax.random.PRNGKey(4), (6, 8))
    out = edit.apply(params, h)
    # R h' = W h + b, and h' - h lies in the row space of R.
    np.testing.assert_allclose(np.asarray(out @ r.T),
                               np.asarray(h @ params["W"].T + params["b"]), atol=1e-4)
    moved = out - h
    np.testing.assert_allclose(np.asarray(moved @ r.T @ r), np.asarray(moved), atol=1e-4)


def test_bad_options():
    with pytest.raises(ValueError, match="positions"):
        LowRankEdit(positions="near")
    with pytest.raises(ValueError, match="rank"):
        LowRankEdit(rank=20).init(jax.random.PRNGKey(0), 8)


@pytest.mark.parametrize("edit_layer", ["mha_block_2_input", "downres_block_3"])
def test_edit_at_init_leaves_the_forward_unchanged(edit_layer):
    model = make_model(policy=F32)
    x = _onehot(2)
    edit = LowRankEdit(rank=2)
    params = edit.init(jax.random.PRNGKey(0), 16 if "mha" in edit_layer else 12)
    readouts = ["transformer_tower", "up_res_block_3"]
    states, edit_input = _run(model, edit, params, edit_layer, readouts, x)
    plain = site_states(model, x, SITE, readouts)
    for name in readouts:
        _close(states[name], plain[name])
    _close(edit_input, site_states(model, x, SITE, [edit_layer])[edit_layer])


def test_edit_reaches_later_layers_and_not_earlier_ones():
    model = make_model(policy=F32)
    x = _onehot(2)
    edit = LowRankEdit(rank=2)
    params = edit.init(jax.random.PRNGKey(0), 16)
    params["b"] = params["b"] + 1.0
    readouts = ["mha_block_1_input", "mha_block_4_input", "transformer_tower"]
    states, _ = _run(model, edit, params, "mha_block_4_input", readouts, x)
    plain = site_states(model, x, SITE, readouts)
    _close(states["mha_block_1_input"], plain["mha_block_1_input"])
    # a readout at the edited tap sees the edit
    r = edit.rotation(params)
    _close(states["mha_block_4_input"] @ r.T,
           plain["mha_block_4_input"] @ np.asarray(params["W"]).T + 1.0)
    assert not np.allclose(np.asarray(states["transformer_tower"]), plain["transformer_tower"])


def test_site_positions_edit_only_the_site_bins():
    model = make_model(policy=F32)
    x = _onehot(2)
    params = LowRankEdit(rank=2).init(jax.random.PRNGKey(0), 4)
    params["b"] = params["b"] + 1.0
    readouts = ["dna_embedder", "transformer_tower"]
    near, _ = _run(model, LowRankEdit(2, "site"), params, "dna_embedder", readouts, x, window=2)
    every, _ = _run(model, LowRankEdit(2, "all"), params, "dna_embedder", readouts, x, window=2)
    plain = site_states(model, x, SITE, readouts, window=2)
    # The edit is affine, so the mean over the edited bins is the same either way.
    _close(near["dna_embedder"], every["dna_embedder"])
    assert not np.allclose(np.asarray(near["dna_embedder"]), plain["dna_embedder"])
    assert not np.allclose(np.asarray(near["transformer_tower"]),
                           np.asarray(every["transformer_tower"]))


def test_head_only_is_the_plain_forward():
    model = make_model(policy=F32)
    x = _onehot(2)
    states, edit_input = _run(model, HeadOnly(), {}, None, ["transformer_tower"], x)
    assert edit_input is None
    _close(states["transformer_tower"],
           site_states(model, x, SITE, ["transformer_tower"])["transformer_tower"])


def test_remat_gives_the_same_gradients():
    model = make_model(policy=F32)
    x = _onehot(2)
    edit = LowRankEdit(rank=2)
    params = edit.init(jax.random.PRNGKey(0), 16)
    organism = jnp.zeros((2,), jnp.int32)

    def loss(p, remat):
        forward = make_forward(model, edit, "mha_block_2_input", ["up_res_block_6"], SITE,
                               remat=remat)
        states, _ = forward(p, model.params, model.state, x, organism)
        return jnp.sum(states["up_res_block_6"] ** 2)

    plain = jax.jit(jax.grad(lambda p: loss(p, False)))(params)
    remat = jax.jit(jax.grad(lambda p: loss(p, True)))(params)
    for key in params:
        _close(remat[key], plain[key])


def test_wrong_module_name_raises():
    from manylatents.dogma.alphagenome.taps import LayerSpec

    model = make_model(policy=F32)
    edit = LowRankEdit(rank=1)
    bad = LayerSpec("nowhere", "alphagenome/nowhere", 128)
    with pytest.raises(KeyError, match="not reached"):
        _run(model, edit, edit.init(jax.random.PRNGKey(0), 16), bad, ["transformer_tower"],
             _onehot(1))


class _Zero:
    """Sets the tapped activation to zero."""

    positions = "all"

    def apply(self, params, h):
        return jnp.zeros_like(h)


@pytest.mark.parametrize("block", [0, 4])
def test_input_edit_replaces_the_residual_stream(block):
    """Zeroing the residual entering block i removes all sequence dependence after it;
    an edit of the attention block's argument alone would leave the skip path intact."""
    model = make_model(policy=F32)
    tap = f"mha_block_{block}_input"
    readouts = [tap, f"mha_block_{block + 1}_input", "transformer_tower"]
    a, edit_input_a = _run(model, _Zero(), {}, tap, readouts, _onehot(1, seed=0))
    b, edit_input_b = _run(model, _Zero(), {}, tap, readouts, _onehot(1, seed=1))
    assert float(jnp.abs(a[tap]).max()) == 0.0
    for name in readouts[1:]:
        np.testing.assert_array_equal(np.asarray(a[name]), np.asarray(b[name]))
    # the edit's input is the unedited residual, which does depend on the sequence
    assert not np.allclose(np.asarray(edit_input_a), np.asarray(edit_input_b))
    _close(edit_input_a, site_states(model, _onehot(1, seed=0), SITE, [tap])[tap])


def test_edit_point_of_tower_inputs():
    from manylatents.dogma.alphagenome.interventions import edit_point
    from manylatents.dogma.alphagenome.taps import LAYERS

    assert edit_point(LAYERS["mha_block_0_input"]) == ("alphagenome/transformer_tower", "input")
    assert edit_point(LAYERS["mha_block_1_input"]) == (
        "alphagenome/transformer_tower/mlp_block", "residual")
    assert edit_point(LAYERS["mha_block_8_input"]) == (
        "alphagenome/transformer_tower/mlp_block_7", "residual")
    assert edit_point(LAYERS["downres_block_3"]) == (LAYERS["downres_block_3"].module, "output")
