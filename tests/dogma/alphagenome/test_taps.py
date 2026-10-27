"""Layer taps on a tiny Haiku model named and laid out like AlphaGenome."""
import numpy as np
import pytest

pytest.importorskip("haiku")
pytest.importorskip("jmp")

from manylatents.dogma.alphagenome.taps import (  # noqa: E402
    DEFAULT_TRUNK_LAYERS, LAYERS, LayerSpec, available_layers, resolve_layers, site_bin,
    site_states,
)
from tests.dogma.alphagenome.fake_model import LENGTH, make_model  # noqa: E402


def _onehot(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.eye(4, dtype=np.float32)[rng.integers(0, 4, (n, LENGTH))]


def test_default_layers():
    assert len(DEFAULT_TRUNK_LAYERS) == 26
    names = [spec.name for spec in DEFAULT_TRUNK_LAYERS]
    assert names[0] == "dna_embedder" and names[-1] == "output_embedder_1"
    assert [spec.resolution for spec in DEFAULT_TRUNK_LAYERS] == (
        [1, 2, 4, 8, 16, 32, 64] + [128] * 10 + [64, 32, 16, 8, 4, 2, 1] + [128, 1])
    assert LAYERS["mha_block_0_input"].module == "alphagenome/transformer_tower/mha_block"
    assert LAYERS["mha_block_8_input"].module == "alphagenome/transformer_tower/mha_block_8"
    assert all(LAYERS[f"mha_block_{i}_input"].tap == "input" for i in range(9))
    assert LAYERS["up_res_block_0"].module == "alphagenome/sequence_decoder/up_res_block"
    assert LAYERS["up_res_block_6"].module == "alphagenome/sequence_decoder/up_res_block_6"


def test_site_bins_at_one_megabase():
    index = 1_048_576 // 2 - 1
    assert index == 524_287
    assert site_bin(index, 128) == 4_095
    assert site_bin(index, 2048) == 255
    assert [site_bin(index, spec.resolution) for spec in DEFAULT_TRUNK_LAYERS[:7]] == [
        524_287, 262_143, 131_071, 65_535, 32_767, 16_383, 8_191]


def test_available_layers_in_forward_order():
    model = make_model()
    seen = available_layers(model, LENGTH)
    order = list(seen)
    positions = [order.index(spec.module) for spec in DEFAULT_TRUNK_LAYERS]
    # The tower module is entered before its blocks; every other tap is in forward order.
    tower = [spec.name for spec in DEFAULT_TRUNK_LAYERS].index("transformer_tower")
    without_tower = positions[:tower] + positions[tower + 1:]
    assert without_tower == sorted(without_tower)
    assert positions[tower] < positions[tower - 9]
    for spec in DEFAULT_TRUNK_LAYERS:
        if spec.tap == "output":
            assert seen[spec.module][1] * spec.resolution == LENGTH, spec


def test_resolve_layers():
    assert resolve_layers(["dna_embedder"]) == (LAYERS["dna_embedder"],)
    custom = LayerSpec("tower_out", "alphagenome/transformer_tower", 128)
    assert resolve_layers([custom]) == (custom,)
    with pytest.raises(KeyError):
        resolve_layers(["nope"])
    with pytest.raises(ValueError):
        resolve_layers(["dna_embedder", "dna_embedder"])
    with pytest.raises(ValueError):
        resolve_layers([])


def _full_activations(model, onehot, specs):
    """Full-length activations of the tapped modules, cast to float32."""
    import haiku as hk
    import jax
    import jax.numpy as jnp

    def apply(params, state, x, org):
        caught = {}

        def interceptor(next_f, args, kwargs, context):
            name = context.module.module_name if context.method_name == "__call__" else None
            for spec in specs:
                if spec.module == name and spec.tap == "input":
                    caught[spec.name] = args[0].astype(jnp.float32)
            out = next_f(*args, **kwargs)
            for spec in specs:
                if spec.module == name and spec.tap == "output":
                    first = out[0] if isinstance(out, tuple) else out
                    caught[spec.name] = first.astype(jnp.float32)
            return out

        with hk.intercept_methods(interceptor):
            model.forward.apply(params, state, None, x, org)
        return caught

    org = jnp.zeros((onehot.shape[0],), jnp.int32)
    return {k: np.asarray(v) for k, v in jax.jit(apply)(model.params, model.state, onehot, org).items()}


@pytest.mark.parametrize("site_index", [0, LENGTH // 2 - 1, LENGTH - 1])
@pytest.mark.parametrize("window", [0, 1])
def test_site_states_equal_slicing_the_full_activation(site_index, window):
    model = make_model(policy="params=float32,compute=float32,output=float32")
    onehot = _onehot(3)
    states = site_states(model, onehot, site_index, window=window)
    full = _full_activations(model, onehot, DEFAULT_TRUNK_LAYERS)
    assert list(states) == [spec.name for spec in DEFAULT_TRUNK_LAYERS]
    for spec in DEFAULT_TRUNK_LAYERS:
        activation = full[spec.name]
        centre = site_index // spec.resolution
        lo, hi = max(0, centre - window), min(activation.shape[1], centre + window + 1)
        expected = activation[:, lo:hi].mean(axis=1)
        assert states[spec.name].dtype == np.float32
        assert states[spec.name].shape == (3, activation.shape[-1])
        np.testing.assert_allclose(states[spec.name], expected, rtol=1e-5, atol=1e-6,
                                   err_msg=spec.name)


def test_tower_input_holds_the_organism_embedding():
    model = make_model(policy="params=float32,compute=float32,output=float32")
    onehot = _onehot(1)
    human = site_states(model, onehot, 10, ["mha_block_0_input", "downres_block_5"])
    mouse = site_states(make_model(policy="params=float32,compute=float32,output=float32",
                                   organism="MUS_MUSCULUS"), onehot, 10,
                        ["mha_block_0_input", "downres_block_5"])
    np.testing.assert_array_equal(human["downres_block_5"], mouse["downres_block_5"])
    assert not np.allclose(human["mha_block_0_input"], mouse["mha_block_0_input"])


def test_bfloat16_compute_returns_float32():
    model = make_model()
    states = site_states(model, _onehot(2), LENGTH // 2 - 1, ["transformer_tower"])
    assert states["transformer_tower"].dtype == np.float32
    assert np.isfinite(states["transformer_tower"]).all()


def test_unreached_module_and_wrong_resolution():
    model = make_model()
    with pytest.raises(KeyError, match="not reached"):
        site_states(model, _onehot(1), 0, [LayerSpec("x", "alphagenome/no_such_block", 1)])
    with pytest.raises(ValueError, match="bp per position"):
        site_states(model, _onehot(1), 0,
                    [LayerSpec("x", "alphagenome/transformer_tower", 64)])
    with pytest.raises(ValueError, match="outside"):
        site_states(model, _onehot(1), LENGTH, ["dna_embedder"])
