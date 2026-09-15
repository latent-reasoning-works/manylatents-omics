"""Correctness tests for Evo2Encoder, the source of the genomic arm's operators.

Pooled per-sequence embeddings become a row-normalised Gaussian operator
P = D^-1 K over pairwise distances, so two properties matter more than usual:
the pooled vectors must be trustworthy points, and the two pooling paths must
agree, because teacher and student must see the same functional.
"""

import pytest
import torch

from manylatents.dogma.encoders.evo2 import Evo2Encoder


def _bare_encoder(layer: str = "blocks.0.mlp.l3", multi: bool = False) -> Evo2Encoder:
    """An encoder with no model loaded: these tests are about pooling and config."""
    encoder = Evo2Encoder.__new__(Evo2Encoder)
    encoder._multi_layer = multi
    encoder._layer_names = [layer]
    encoder.device = "cpu"
    return encoder


def test_pooling_paths_agree_in_float32():
    """encode() stayed bfloat16 while the batched path promoted to float32.

    Two points 0.0026 apart in fp32 collapse to exactly 0.0 in bf16, and the
    operator is exp(-d^2 / 2 sigma^2) over cdist of these vectors, so bf16
    pooling destroys the near-neighbour structure the operator is made of.
    """
    hidden = torch.randn(1, 8, 16, dtype=torch.bfloat16)
    encoder = _bare_encoder()
    single = encoder._pool_embeddings({"blocks.0.mlp.l3": hidden})
    batched = encoder._pool_embeddings(
        {"blocks.0.mlp.l3": hidden}, mask=torch.ones(1, 8, dtype=torch.bool)
    )
    assert single.dtype == torch.float32
    assert batched.dtype == torch.float32
    assert torch.allclose(single, batched, atol=1e-6)


def test_masked_mean_ignores_padding():
    """Padding must not drag the pooled point toward the origin."""
    hidden = torch.ones(1, 4, 3)
    hidden[:, 2:] = 99.0
    mask = torch.tensor([[True, True, False, False]])
    pooled = _bare_encoder()._pool_embeddings({"blocks.0.mlp.l3": hidden}, mask=mask)
    assert torch.allclose(pooled, torch.ones(1, 3))


def test_every_model_size_declares_default_layers():
    """Only evo2_1b_base had default_layers, so the 1B returned a dict of three
    tensors and 7B/40B returned a single tensor. A tensor-based operator builder
    breaks on the dict."""
    for name, config in Evo2Encoder.MODELS.items():
        assert "default_layers" in config, name
        assert "default_layer" in config, name


def test_fortyb_middle_layer_is_actually_middle():
    """Upstream depths are 25 / 32 / 50 blocks, so blocks.32 for the 40B is the
    33rd of 50 -- 66%, not the middle it was labelled."""
    assert Evo2Encoder.MODELS["evo2_40b"]["default_layer"] == "blocks.25.mlp.l3"
    assert Evo2Encoder.MODELS["evo2_40b_base"]["default_layer"] == "blocks.25.mlp.l3"


def test_no_layer_default_is_the_final_block():
    """Evo2-7B build-and-flush: informative geometry at intermediate layers,
    numerical annihilation at the final one. No default may sit at the end."""
    depths = {"evo2_1b_base": 25, "evo2_7b": 32, "evo2_7b_base": 32,
              "evo2_40b": 50, "evo2_40b_base": 50}
    for name, config in Evo2Encoder.MODELS.items():
        for layer in config["default_layers"]:
            index = int(layer.split(".")[1])
            assert index < depths[name] - 1, f"{name}: {layer} is at or past the last block"


def test_padding_uses_the_tokenizer_pad_id():
    """_tokenize_batch padded with 0, which is EOS/EOD; the pad id is 1."""
    encoder = _bare_encoder()

    class FakeTokenizer:
        pad_id = 1

        def tokenize(self, seq):
            return [4] * len(seq)

    class FakeModel:
        tokenizer = FakeTokenizer()

    encoder._model = FakeModel()
    encoder._ensure_loaded = lambda: None
    batch = encoder._tokenize_batch(["ACGT", "AC"])
    assert batch["input_ids"][1, 2].item() == 1
    assert batch["input_ids"][1, 3].item() == 1
    assert batch["attention_mask"][1].tolist() == [True, True, False, False]


def test_dead_weights_constant_is_gone():
    """DEFAULT_WEIGHTS was never read: _load_model called Evo2(self.model_name),
    which resolves by name. It is replaced by an explicit weights_path."""
    assert not hasattr(Evo2Encoder, "DEFAULT_WEIGHTS")


def test_weights_path_is_accepted_and_recorded():
    """Offline clusters cannot download by name, so the path must be settable."""
    encoder = Evo2Encoder(model_name="evo2_1b_base", layer_name="blocks.14.mlp.l3",
                          weights_path="/weights/evo2_1b_base.pt", device="cpu")
    assert encoder.weights_path == "/weights/evo2_1b_base.pt"


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="model_name"):
        Evo2Encoder(model_name="evo2_3b", device="cpu")
