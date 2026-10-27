"""Trainable edits of AlphaGenome's hidden states, applied inside the frozen forward.

An intervention changes the activation at one tap (a :class:`~.taps.LayerSpec`)
while every AlphaGenome parameter stays fixed; later layers see the edited
activation. :func:`make_forward` builds that forward with
``hk.intercept_methods`` and reduces the readout layers to the site's bin, as
:func:`~.taps.site_states` does, so only ``[B, D]`` per layer leaves it.

Interventions are plain classes with ``init(rng, dim) -> params`` and
``apply(params, h) -> h'`` on the last axis of ``h``; their parameters are a
pytree that a :class:`~.train.JaxTrainer` optimises. No torch here.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence, Union

from manylatents.dogma.alphagenome.taps import (
    _TOWER, LayerSpec, _numbered, resolve_layers, site_bin,
)

POSITIONS = ("all", "site")
# Modules rematerialised when remat=True: the blocks of the encoder, tower and
# decoder, by the last component of their Haiku name.
REMAT_PATTERN = r"(downres_block|mha_block|mlp_block|up_res_block)(_\d+)?$"


def _orthonormal_rows(a):
    """``[r, d]`` with orthonormal rows: Q of the reduced QR of ``a`` (``[d, r]``),
    signs fixed so that ``R``'s diagonal is positive (QR is then unique and smooth)."""
    import jax.numpy as jnp

    q, r = jnp.linalg.qr(a)
    signs = jnp.where(jnp.diagonal(r) < 0, -1.0, 1.0)
    return (q * signs).T


class LowRankEdit:
    """LoReFT-style low-rank edit (Wu et al. 2024): ``h + R^T (W h + b - R h)``.

    ``R`` (``r x d``) has orthonormal rows, so the edit acts only inside the
    r-dimensional subspace that ``R`` spans and sets ``R h'`` to ``W h + b``.
    ``R`` is parametrised as the Q factor of an unconstrained ``d x r``
    matrix ``A`` (sign-fixed QR), so it is orthonormal at every step without a
    penalty or a projection; the optimiser updates ``A``. ``W`` starts equal to
    ``R`` and ``b`` at zero, so the edit is exactly the identity at step 0.

    Args:
        rank: r.
        positions: ``all`` edits every position of the tapped activation;
            ``site`` edits only the bins that :func:`make_forward` reads at the
            site (the site's bin, plus ``window`` on each side).
    """

    name = "low_rank_edit"

    def __init__(self, rank: int = 8, positions: str = "all"):
        if positions not in POSITIONS:
            raise ValueError(f"positions must be one of {POSITIONS}, got {positions!r}")
        if int(rank) < 1:
            raise ValueError("rank must be >= 1")
        self.rank = int(rank)
        self.positions = positions

    def init(self, rng, dim: int) -> dict:
        import jax
        import jax.numpy as jnp

        if self.rank > dim:
            raise ValueError(f"rank {self.rank} exceeds the layer width {dim}")
        a = jax.random.normal(rng, (dim, self.rank), jnp.float32)
        return {"A": a, "W": _orthonormal_rows(a), "b": jnp.zeros((self.rank,), jnp.float32)}

    def rotation(self, params):
        """``R``, ``[r, d]`` with orthonormal rows."""
        return _orthonormal_rows(params["A"])

    def apply(self, params, h):
        import jax.numpy as jnp

        r = self.rotation(params)
        x = h.astype(jnp.float32)
        edit = (x @ params["W"].T + params["b"] - x @ r.T) @ r
        return (x + edit).astype(h.dtype)

    def jacobian_sq_norm(self, params, residual: bool = False):
        """``||J||_F^2`` of the edit, in closed form; the same at every input.

        The edit is affine, ``h' = h + R^T (W - R) h + R^T b``, so ``J = I +
        R^T M`` with ``M = W - R``, and since ``R R^T = I_r``:
        ``||R^T M||_F^2 = ||M||_F^2`` (``residual``: the Jacobian of ``h' - h``)
        and ``||I + R^T M||_F^2 = d + 2 tr(R^T M) + ||M||_F^2``.
        """
        import jax.numpy as jnp

        r = self.rotation(params)
        m = params["W"] - r
        value = jnp.sum(m ** 2)
        if residual:
            return value
        return r.shape[1] + 2 * jnp.sum(r * m) + value

    def __repr__(self):
        return f"LowRankEdit(rank={self.rank}, positions={self.positions!r})"


class HeadOnly:
    """No edit: AlphaGenome's states stay frozen and only an objective's head
    (e.g. :class:`~.objectives.TrackHead`) is trained, on one or more readout layers.

    With nothing to differentiate through the model, the site states are
    computed once and cached (see :class:`~.finetune.AlphaGenomeFinetune`).
    """

    name = "head_only"
    positions = "site"

    def init(self, rng, dim: int) -> dict:
        return {}

    def apply(self, params, h):
        return h

    def __repr__(self):
        return "HeadOnly()"


def _first(x):
    return x[0] if isinstance(x, (tuple, list)) and x else x


def _replace_first(out, new):
    if isinstance(out, (tuple, list)) and out:
        return type(out)((new, *out[1:]))
    return new


def _site_slice(spec: LayerSpec, length: int, site_index: int, window: int):
    centre = site_bin(site_index, spec.resolution)
    n = length // spec.resolution
    return max(0, centre - window), min(n, centre + window + 1)


_TOWER_MHA = re.compile(re.escape(_TOWER) + r"/mha_block(?:_(\d+))?$")


def edit_point(spec: LayerSpec) -> tuple[str, str]:
    """``(module, how)``: where :func:`make_forward` applies an edit at ``spec``.

    The tower's loop is ``x += MHABlock()(x, bias); x += MLPBlock()(x)``, with
    the pair update of every second block also reading ``x``. Editing the
    first argument of ``mha_block_i`` would change only what that attention
    block sees, while the residual ``x`` went on unedited. An edit at
    ``mha_block_{i}_input`` therefore edits the residual stream entering block
    ``i``, so that every later reader sees it:

    * ``i = 0``: the input of ``transformer_tower`` (``how='input'``);
    * ``i > 0``: the output of the previous block's ``mlp_block``, which is
      added last before block ``i``: that module returns
      ``edit(x + out) - x`` for the residual ``x`` it received
      (``how='residual'``), so the tower's ``x + ...`` gives ``edit(x + out)``.

    Any other tap is edited where it is read: the module's output
    (``how='output'``), or its first positional argument (``how='input'``,
    which changes only what that one module sees).
    """
    match = _TOWER_MHA.match(spec.module) if spec.tap == "input" else None
    if match is None:
        return spec.module, spec.tap
    i = int(match.group(1) or 0)
    if i == 0:
        return _TOWER, "input"
    return f"{_TOWER}/{_numbered('mlp_block', i - 1)}", "residual"


def _remat_call(next_f, args, kwargs):
    """``next_f(*args, **kwargs)`` under ``hk.remat``; only array leaves are
    checkpoint inputs, everything else (flags, None, ints) is closed over."""
    import haiku as hk
    import jax

    leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
    is_array = [isinstance(leaf, jax.Array) for leaf in leaves]

    def call(*arrays):
        it = iter(arrays)
        full = [next(it) if flag else leaf for leaf, flag in zip(leaves, is_array)]
        a, k = jax.tree_util.tree_unflatten(treedef, full)
        return next_f(*a, **k)

    return hk.remat(call)(*[leaf for leaf, flag in zip(leaves, is_array) if flag])


def make_forward(model, intervention, edit_layer: Optional[Union[str, LayerSpec]],
                 readout_layers: Sequence[Union[str, LayerSpec]], site_index: int,
                 window: int = 0, remat: bool = False):
    """The frozen AlphaGenome forward with ``intervention`` at ``edit_layer``.

    Returns ``apply(edit_params, params, state, dna_sequence, organism_index)
    -> (states, edit_input)``: ``states`` maps each readout layer to its
    ``[B, D]`` float32 state at the site (mean over ``window`` bins on each
    side), after the edit; ``edit_input`` is the ``[B, D]`` float32
    activation at the edit layer's site bin before the edit (``None`` without
    an edit layer), which :class:`~.objectives.JacobianPenalty` evaluates at.

    Where the edit goes is :func:`edit_point`: a module's output for
    ``tap='output'``; for ``mha_block_{i}_input`` the residual stream entering
    tower block ``i`` (so the pair update, the attention block, the MLP block
    and the skip connection all see the edited stream); and the first
    positional argument of the module for any other ``tap='input'``. A readout
    at the edited tap sees the edited activation. Under a bfloat16 policy the
    residual edit is exact up to one bfloat16 rounding (the tower adds
    ``edit(r) - x`` back to ``x``). Not jitted: the trainer jits the whole
    step. With ``remat``, every encoder, tower and decoder block
    (:data:`REMAT_PATTERN`) is rematerialised in the backward pass.
    """
    import haiku as hk
    import jax.numpy as jnp

    from manylatents.dogma.alphagenome.taps import _reduce_to_site

    readouts = resolve_layers(readout_layers)
    edit_spec = resolve_layers([edit_layer])[0] if edit_layer is not None else None
    edit_module, edit_how = edit_point(edit_spec) if edit_spec is not None else (None, None)
    by_module: dict[str, list[LayerSpec]] = {}
    for spec in readouts:
        by_module.setdefault(spec.module, []).append(spec)
    remat_re = re.compile(REMAT_PATTERN)
    edit_site = getattr(intervention, "positions", "all") == "site"

    def apply(edit_params, params, state, dna_sequence, organism_index):
        length = dna_sequence.shape[1]
        caught = {}

        def edit(x):
            caught["__edit_input__"] = _reduce_to_site(x, edit_spec, length, site_index, window)
            if edit_site:
                lo, hi = _site_slice(edit_spec, length, site_index, window)
                return x.at[:, lo:hi].set(intervention.apply(edit_params, x[:, lo:hi]))
            return intervention.apply(edit_params, x)

        def interceptor(next_f, args, kwargs, context):
            if context.method_name != "__call__":
                return next_f(*args, **kwargs)
            name = context.module.module_name
            wanted = by_module.get(name, ())
            editing = edit_module == name
            if editing and edit_how == "input":
                args = (edit(args[0]), *args[1:])
            for spec in wanted:
                if spec.tap == "input":
                    caught[spec.name] = _reduce_to_site(args[0], spec, length, site_index, window)
            if remat and remat_re.search(name.rsplit("/", 1)[-1]):
                out = _remat_call(next_f, args, kwargs)
            else:
                out = next_f(*args, **kwargs)
            if editing and edit_how == "output":
                out = _replace_first(out, edit(_first(out)))
            elif editing and edit_how == "residual":
                residual = args[0]
                added = _first(out)
                edited = edit(residual + added)
                out = _replace_first(out, (edited - residual).astype(added.dtype))
            for spec in wanted:
                if spec.tap == "output":
                    caught[spec.name] = _reduce_to_site(_first(out), spec, length, site_index,
                                                        window)
            return out

        with hk.intercept_methods(interceptor):
            model.forward.apply(params, state, None, dna_sequence, organism_index)
        missing = [spec.module for spec in readouts if spec.name not in caught]
        if edit_spec is not None and "__edit_input__" not in caught:
            missing.append(edit_module)
        if missing:
            raise KeyError(f"modules not reached during the forward pass: {missing}; "
                           "see taps.available_layers()")
        edit_input = caught.pop("__edit_input__", None)
        return {spec.name: caught[spec.name].astype(jnp.float32) for spec in readouts}, edit_input

    return apply
