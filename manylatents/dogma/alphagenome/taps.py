"""Hidden states of AlphaGenome at one site, layer by layer.

AlphaGenome is Haiku, so there are no forward hooks; ``hk.intercept_methods``
sees every module call instead. A tap names a module and whether it reads that
module's output or its first input. :func:`site_states` reduces each tapped
activation to the sequence bin that holds the site inside the interceptor, so
only ``[B, D]`` per layer leaves the compiled function, never a full-length
activation.

Resolutions follow ``alphagenome_research.model.model`` (0.2.0): the encoder
halves the length after the DNA embedder and after each of six down-resolution
blocks (2 to 64 bp), the transformer tower runs at 128 bp, and the decoder's
seven up-resolution blocks return from 64 bp to 1 bp.
"""
from __future__ import annotations

import functools
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence, Union

import numpy as np

_ROOT = "alphagenome"
_ENCODER = f"{_ROOT}/sequence_encoder"
_TOWER = f"{_ROOT}/transformer_tower"
_DECODER = f"{_ROOT}/sequence_decoder"


@dataclass(frozen=True)
class LayerSpec:
    """One tap.

    Attributes:
        name: what the tap is called in outputs.
        module: full Haiku module name.
        resolution: bp per position of the tapped activation.
        tap: ``output`` (the module's result; the first element of a tuple) or
            ``input`` (the first positional argument of its ``__call__``).
    """

    name: str
    module: str
    resolution: int
    tap: str = "output"


def _numbered(base: str, i: int) -> str:
    """Haiku names unnamed modules made in a loop ``base``, ``base_1``, ``base_2``, ..."""
    return base if i == 0 else f"{base}_{i}"


# Forward order. The residual stream entering each of the nine attention
# blocks; the first already holds the organism embedding (model.py adds it to
# the trunk before the tower).
DEFAULT_TRUNK_LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec("dna_embedder", f"{_ENCODER}/dna_embedder", 1),
    *(LayerSpec(f"downres_block_{i}", f"{_ENCODER}/downres_block_{i}", 2 ** (i + 1))
      for i in range(6)),
    *(LayerSpec(f"mha_block_{i}_input", f"{_TOWER}/{_numbered('mha_block', i)}", 128, "input")
      for i in range(9)),
    LayerSpec("transformer_tower", _TOWER, 128),
    *(LayerSpec(f"up_res_block_{i}", f"{_DECODER}/{_numbered('up_res_block', i)}", 64 >> i)
      for i in range(7)),
    LayerSpec("output_embedder", f"{_ROOT}/output_embedder", 128),
    LayerSpec("output_embedder_1", f"{_ROOT}/output_embedder_1", 1),
)
LAYERS = {spec.name: spec for spec in DEFAULT_TRUNK_LAYERS}


def resolve_layers(layers: Iterable[Union[str, LayerSpec]]) -> tuple[LayerSpec, ...]:
    """LayerSpecs for names in ``LAYERS`` (or LayerSpecs, kept as given)."""
    out = []
    for layer in layers:
        if isinstance(layer, LayerSpec):
            out.append(layer)
        elif layer in LAYERS:
            out.append(LAYERS[layer])
        else:
            raise KeyError(f"unknown layer {layer!r}; known: {list(LAYERS)}")
    names = [spec.name for spec in out]
    if not out or len(set(names)) != len(names):
        raise ValueError(f"layers must be non-empty and distinct, got {names}")
    return tuple(out)


def _first(x):
    return x[0] if isinstance(x, (tuple, list)) and x else x


def available_layers(model, length: int) -> "OrderedDict[str, tuple]":
    """Every module the forward calls, in the order of their first call, with
    its output shape.

    Traces with ``jax.eval_shape``: no FLOPs, no device memory. A parent comes
    before the modules it calls. A module that returns a tuple reports its
    first element; one that returns no array reports ``()``.
    """
    import haiku as hk
    import jax
    import jax.numpy as jnp

    seen: "OrderedDict[str, tuple]" = OrderedDict()

    def interceptor(next_f, args, kwargs, context):
        if context.method_name != "__call__":
            return next_f(*args, **kwargs)
        name = context.module.module_name
        first = name not in seen
        if first:
            seen[name] = ()
        out = next_f(*args, **kwargs)
        if first:
            seen[name] = tuple(getattr(_first(out), "shape", ()))
        return out

    def traced(params, state, dna_sequence, organism_index):
        with hk.intercept_methods(interceptor):
            model.forward.apply(params, state, None, dna_sequence, organism_index)

    jax.eval_shape(traced, model.params, model.state,
                   jax.ShapeDtypeStruct((1, int(length), 4), jnp.float32),
                   jax.ShapeDtypeStruct((1,), jnp.int32))
    return seen


def site_bin(site_index: int, resolution: int) -> int:
    """Index of the bin holding ``site_index`` at ``resolution`` bp per bin."""
    return int(site_index) // int(resolution)


def _reduce_to_site(x, spec: LayerSpec, length: int, site_index: int, window: int):
    """``[B, D]`` float32: mean of bins ``bin - window .. bin + window`` (clipped)."""
    import jax.numpy as jnp

    if x.ndim != 3 or x.shape[1] * spec.resolution != length:
        raise ValueError(
            f"layer {spec.name!r} ({spec.module}) has shape {tuple(x.shape)}; expected "
            f"[B, {length // spec.resolution}, D] at {spec.resolution} bp per position"
        )
    centre = site_bin(site_index, spec.resolution)
    lo, hi = max(0, centre - window), min(x.shape[1], centre + window + 1)
    return jnp.mean(x[:, lo:hi].astype(jnp.float32), axis=1)


@functools.lru_cache(maxsize=16)
def _compiled(model, specs: tuple, site_index: int, window: int):
    import haiku as hk
    import jax

    by_module: dict[str, list[LayerSpec]] = {}
    for spec in specs:
        by_module.setdefault(spec.module, []).append(spec)

    def apply(params, state, dna_sequence, organism_index):
        length = dna_sequence.shape[1]
        caught = {}

        def interceptor(next_f, args, kwargs, context):
            wanted = (by_module.get(context.module.module_name, ())
                      if context.method_name == "__call__" else ())
            for spec in wanted:
                if spec.tap == "input":
                    caught[spec.name] = _reduce_to_site(args[0], spec, length, site_index, window)
            out = next_f(*args, **kwargs)
            for spec in wanted:
                if spec.tap == "output":
                    caught[spec.name] = _reduce_to_site(_first(out), spec, length, site_index,
                                                        window)
            return out

        with hk.intercept_methods(interceptor):
            # Predictions are discarded; XLA drops whatever no tap depends on.
            model.forward.apply(params, state, None, dna_sequence, organism_index)
        return caught

    return jax.jit(apply)


def site_states(model, onehot_batch, site_index: int,
                layers: Sequence[Union[str, LayerSpec]] = DEFAULT_TRUNK_LAYERS,
                window: int = 0) -> dict:
    """Hidden state at the site for each layer: name -> ``[B, D]`` float32 numpy.

    Args:
        model: a :class:`~manylatents.dogma.alphagenome.model.LocalModel`.
        onehot_batch: ``[B, L, 4]`` one-hot sequences.
        site_index: 0-based position of the site in the window.
        layers: names in ``LAYERS`` or LayerSpecs; compiled once per set.
        window: also average this many bins on each side of the site's bin.

    Raises:
        KeyError: a tapped module was never called (wrong module name).
        ValueError: a tapped activation does not have the layer's resolution.
    """
    import jax.numpy as jnp

    specs = resolve_layers(layers)
    onehot_batch = jnp.asarray(onehot_batch)
    if onehot_batch.ndim != 3 or onehot_batch.shape[-1] != 4:
        raise ValueError(f"onehot_batch must be [B, L, 4], got {tuple(onehot_batch.shape)}")
    if not 0 <= int(site_index) < onehot_batch.shape[1]:
        raise ValueError(f"site_index {site_index} outside length {onehot_batch.shape[1]}")
    if int(window) < 0:
        raise ValueError("window must be >= 0")
    organism = jnp.full((onehot_batch.shape[0],), model.organism_index, dtype=jnp.int32)
    caught = _compiled(model, specs, int(site_index), int(window))(
        model.params, model.state, onehot_batch, organism)
    missing = [(spec.name, spec.module) for spec in specs if spec.name not in caught]
    if missing:
        raise KeyError(f"modules not reached during the forward pass: {missing}; "
                       "see available_layers()")
    return {spec.name: np.asarray(caught[spec.name], dtype=np.float32) for spec in specs}
