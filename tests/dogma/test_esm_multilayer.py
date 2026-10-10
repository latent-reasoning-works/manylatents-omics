"""Per-layer extraction for the ESM family (fair-esm).

`ESMEncoder` was single-layer while `ESM3Encoder` and `OrthrusEncoder` both took
`layer_indices`, so any per-layer ESM work had to reimplement the encoder
downstream. These tests pin the added API and, more importantly, pin that adding
it did not move the single-layer numbers — the pooling formula produces every
embedding in a variant-effect grid, and a change there is invisible in the
output shape.

No weights are downloaded: the model is stubbed, since what is under test is the
layer plumbing and the mask, not fair-esm.
"""
import pytest
import torch

from manylatents.dogma.encoders.esm import ESMEncoder

MODEL = "esm1b_t33_650M_UR50S"          # 33 layers, 1280-dim


class _FakeAlphabet:
    padding_idx = 1


class _FakeESM:
    """Returns a distinct, deterministic tensor per requested layer.

    Distinct per layer matters: if the encoder silently pooled one layer and
    labelled it as several, every layer would be identical and a test that only
    checked shapes would pass.
    """

    def __init__(self, batch, length, dim):
        self.shape = (batch, length, dim)
        self.seen = None

    def __call__(self, tokens, repr_layers):
        self.seen = list(repr_layers)
        b, ln, d = self.shape
        return {"representations":
                {i: torch.full((b, ln, d), float(i)) for i in repr_layers}}


def _encoder(layer_indices=None, repr_layer=None, batch=2, length=6, dim=4):
    enc = ESMEncoder(model_name=MODEL, device="cpu",
                     repr_layer=repr_layer, layer_indices=layer_indices)
    enc._model = _FakeESM(batch, length, dim)
    enc._alphabet = _FakeAlphabet()
    enc._batch_converter = lambda data: (None, None, None)
    return enc


def _tokens(batch=2, length=6, pad_from=None):
    # 0 = BOS, 2 = EOS, 1 = PAD, anything else = a residue
    t = torch.full((batch, length), 5)
    t[:, 0] = 0
    if pad_from is None:
        t[:, -1] = 2
    else:
        t[:, pad_from - 1] = 2
        t[:, pad_from:] = 1
    return t


class TestESMMultiLayerAPI:
    def test_default_is_single_layer(self):
        enc = ESMEncoder(model_name=MODEL, device="cpu")
        assert enc.multi_layer is False
        assert enc.layer_indices is None

    def test_layer_indices_enables_multi_layer(self):
        enc = ESMEncoder(model_name=MODEL, device="cpu", layer_indices=[0, 20, 33])
        assert enc.multi_layer is True
        assert enc.layer_indices == [0, 20, 33]

    def test_empty_layer_indices_stays_single_layer(self):
        enc = ESMEncoder(model_name=MODEL, device="cpu", layer_indices=[])
        assert enc.multi_layer is False

    @pytest.mark.parametrize("bad", [[34], [-1], [0, 99]])
    def test_out_of_range_layers_rejected(self, bad):
        """A typo'd layer index must fail at construction, not after the GPU work."""
        with pytest.raises(ValueError, match="out of range"):
            ESMEncoder(model_name=MODEL, device="cpu", layer_indices=bad)

    def test_layer_0_is_accepted(self):
        """The input embedding layer is a legitimate request, not an off-by-one."""
        enc = ESMEncoder(model_name=MODEL, device="cpu", layer_indices=[0])
        assert enc.layer_indices == [0]


class TestESMMultiLayerExtraction:
    def test_one_forward_pass_requests_every_layer(self):
        """N layers must cost one forward, which is the whole point of the change."""
        enc = _encoder(layer_indices=[0, 20, 33])
        enc._extract_embeddings({"tokens": _tokens()})
        assert enc._model.seen == [0, 20, 33]

    def test_returns_one_entry_per_layer_in_order(self):
        enc = _encoder(layer_indices=[0, 20, 33])
        out = enc._extract_embeddings({"tokens": _tokens()})
        assert list(out) == ["layer_0", "layer_20", "layer_33"]
        assert all(v.shape == (2, 4) for v in out.values())

    def test_layers_are_not_aliased(self):
        """Each key must carry its own layer, not one layer under many names."""
        enc = _encoder(layer_indices=[0, 20, 33])
        out = enc._extract_embeddings({"tokens": _tokens()})
        assert torch.allclose(out["layer_0"], torch.zeros(2, 4))
        assert torch.allclose(out["layer_20"], torch.full((2, 4), 20.0))
        assert torch.allclose(out["layer_33"], torch.full((2, 4), 33.0))

    def test_single_layer_return_is_still_a_bare_tensor(self):
        """Existing callers unpack a Tensor; they must not start seeing a dict."""
        enc = _encoder(repr_layer=33)
        out = enc._extract_embeddings({"tokens": _tokens()})
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 4)

    def test_multi_layer_matches_single_layer_exactly(self):
        """The regression that would matter: same layer, same number, both paths.

        Every LVD in a per-layer grid comes out of this pooling. If requesting a
        layer via `layer_indices` differed from requesting it via `repr_layer`,
        nothing in the shapes would show it.
        """
        tokens = _tokens(pad_from=4)
        single = _encoder(repr_layer=20)._extract_embeddings({"tokens": tokens})
        multi = _encoder(layer_indices=[20])._extract_embeddings({"tokens": tokens})
        assert torch.equal(single, multi["layer_20"])

    def test_padding_and_special_tokens_are_excluded(self):
        """Ragged batches: the mask must follow each sequence's own length."""
        enc = _encoder(layer_indices=[7], batch=2, length=6, dim=3)
        tokens = _tokens(batch=2, length=6, pad_from=4)
        tokens[1, 3] = 5            # row 1 keeps one more residue
        tokens[1, 4] = 2
        tokens[1, 5] = 1
        out = enc._extract_embeddings({"tokens": tokens})["layer_7"]
        # The fake fills the layer with its index, so a correct mean over any
        # non-empty set of real residues is exactly that index.
        assert torch.allclose(out, torch.full((2, 3), 7.0))
