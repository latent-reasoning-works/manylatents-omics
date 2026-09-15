"""The opt-in trainable path.

Evo 2 is inference-only as shipped: vortex's load_checkpoint runs under
torch.inference_mode(), so every weight is an inference tensor and cannot be
saved for backward whatever requires_grad says. Spikes 465429 and 465441 on
Tamia settled both halves of that -- see docs/evo2_trainability.md.

The GPU tests skip without the evo2 package and CUDA; the de-inference helper
is exercised on CPU with a stand-in module so the logic is covered everywhere.
"""

import importlib.util

import pytest
import torch
from torch import nn

from manylatents.dogma.encoders.evo2 import Evo2Encoder


class _Stub(nn.Module):
    """A module with both a parameter and a buffer, like vortex's filter."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.register_buffer("t", torch.arange(4.0))


def test_dematerialise_converts_parameters_and_buffers():
    """Buffers are not optional: compute_filter evaluates (log_poles * self.t),
    and one inference operand poisons the whole graph."""
    encoder = Evo2Encoder.__new__(Evo2Encoder)
    encoder._model = _Stub()
    params, buffers = encoder._dematerialise_inference_tensors()
    assert params == 1
    assert buffers == 1


def test_dematerialise_preserves_values_and_grad_flags():
    """Re-materialising must not perturb the weights: the frozen path's numbers
    have to be reproducible, since teacher caches depend on them."""
    encoder = Evo2Encoder.__new__(Evo2Encoder)
    stub = _Stub()
    encoder._model = stub
    before_weight = stub.weight.detach().clone()
    before_buffer = stub.t.detach().clone()
    encoder._dematerialise_inference_tensors()
    assert torch.equal(stub.weight.detach(), before_weight)
    assert torch.equal(stub.t, before_buffer)
    assert stub.weight.requires_grad


def test_dematerialise_skips_none_entries():
    """A module with an optional bias set to None must not crash the walk."""
    encoder = Evo2Encoder.__new__(Evo2Encoder)
    module = nn.Linear(3, 3, bias=False)
    assert module.bias is None
    encoder._model = module
    params, buffers = encoder._dematerialise_inference_tensors()
    assert params == 1 and buffers == 0


def test_trainable_defaults_to_false():
    encoder = Evo2Encoder(model_name="evo2_1b_base", layer_name="blocks.14.mlp.l3",
                          device="cpu")
    assert encoder.trainable is False


def test_trainable_is_accepted_and_recorded():
    encoder = Evo2Encoder(model_name="evo2_1b_base", layer_name="blocks.14.mlp.l3",
                          trainable=True, device="cpu")
    assert encoder.trainable is True


# --- the GPU half -------------------------------------------------------------

# A module-level importorskip would skip the CPU tests above too, since it
# raises at collection. Decide per test instead.
_HAVE_EVO2 = importlib.util.find_spec("evo2") is not None
gpu = pytest.mark.skipif(
    not (_HAVE_EVO2 and torch.cuda.is_available()),
    reason="needs the evo2 package, a GPU, and the Evo 2 weights",
)


@gpu
def test_operator_loss_reaches_model_parameters():
    """The measured result from spike 465441: 159 of 265 parameters receive
    finite non-zero gradients. Blocks after the hooked layer correctly get none."""
    encoder = Evo2Encoder(model_name="evo2_1b_base", layer_name="blocks.14.mlp.l3",
                          trainable=True)
    batch = encoder._tokenize_batch(["ACGT" * 32, "TTGA" * 32, "GGCA" * 32, "CTAG" * 32])
    pooled = encoder._extract_embeddings(batch)
    assert pooled.requires_grad

    distance = torch.cdist(pooled, pooled,
                           compute_mode="donot_use_mm_for_euclid_dist")
    off = ~torch.eye(len(pooled), dtype=torch.bool, device=distance.device)
    sigma = distance[off].median().clamp_min(1e-6)
    kernel = torch.exp(-(distance * distance) / (2 * sigma * sigma))
    operator = kernel / kernel.sum(-1, keepdim=True)
    assert (operator.diagonal() > 0).all(), "self-affinity must survive"
    operator.square().sum().backward()

    inner = encoder._model.model
    got = [p for p in inner.parameters()
           if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0]
    assert got, "no parameter received a finite non-zero gradient"


@gpu
def test_frozen_path_is_numerically_untouched():
    """The claim docs/evo2_trainability.md left unverified. Teacher caches are
    built with trainable=False, so cloning must not move the numbers."""
    batch_sequences = ["ACGT" * 32, "TTGA" * 32]
    frozen = Evo2Encoder(model_name="evo2_1b_base", layer_name="blocks.14.mlp.l3")
    reference = frozen._extract_embeddings(frozen._tokenize_batch(batch_sequences))

    cloned = Evo2Encoder(model_name="evo2_1b_base", layer_name="blocks.14.mlp.l3",
                         trainable=True)
    with torch.no_grad():
        candidate = cloned._extract_embeddings(cloned._tokenize_batch(batch_sequences))

    assert torch.allclose(reference, candidate.detach(), atol=1e-5), \
        (reference - candidate.detach()).abs().max()


@gpu
def test_default_path_does_not_require_grad():
    encoder = Evo2Encoder(model_name="evo2_1b_base", layer_name="blocks.14.mlp.l3")
    batch = encoder._tokenize_batch(["ACGT" * 32])
    assert not encoder._extract_embeddings(batch).requires_grad
