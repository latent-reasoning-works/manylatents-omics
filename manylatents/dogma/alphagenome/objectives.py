"""Losses for finetuning or steering AlphaGenome's representations, in pure JAX.

Each objective is a small class with a ``name``, a ``weight``, ``init(rng,
outputs, batch) -> params`` (``{}`` when it has none) and ``__call__(outputs,
batch, params=None) -> scalar``. :class:`Composite` sums weighted objectives.
``outputs`` is what :class:`~.finetune.AlphaGenomeFinetune` computes for a
minibatch of variants:

    outputs["ref"][layer], outputs["alt"][layer], outputs["delta"][layer]
        [B, D] float32 site states (after the edit) and alt - ref
    outputs["edit_input"]   [2B, D] the edit layer's site activation before the edit
    outputs["edit_fn"]      h -> edited h, with the current edit parameters
    outputs["rng"]          a PRNG key for this step

and ``batch`` holds per-variant arrays the caller supplies (``target``,
``teacher``).

Targets and teachers must not come from the benchmark labels used for
evaluation: training on them would make the evaluation circular (the project
pre-registers this). The caller supplies targets; no objective reads labels.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

SOURCES = ("delta", "ref", "alt")
TRACK_LOSSES = ("mse", "poisson", "cosine")


def _source(outputs, source: str, layers: Sequence[str]):
    """``[B, sum D]``: the ``source`` states at ``layers``, concatenated."""
    import jax.numpy as jnp

    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return jnp.concatenate([outputs[source][layer] for layer in layers], axis=-1)


def _as_list(layers) -> list:
    return [layers] if isinstance(layers, str) else list(layers)


class Objective:
    """Base: no parameters, no required batch keys."""

    name = "objective"
    batch_keys: tuple = ()

    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        self.weight = float(weight)
        if name is not None:
            self.name = name

    def init(self, rng, outputs, batch) -> dict:
        return {}

    def __call__(self, outputs, batch, params=None):
        raise NotImplementedError


class TrackHead(Objective):
    """A head from the site state to a target vector per variant.

    Use for MPRA allelic effects, caQTL effects or AlphaGenome track-delta
    vectors. Targets come from the caller (``batch[target_key]``, ``[B, T]``),
    never from evaluation labels (see the module docstring).

    Args:
        layers: readout layer name(s); their states are concatenated.
        source: ``delta`` (alt - ref), ``ref`` or ``alt``. (Not ``on``: YAML 1.1
            reads an unquoted ``on`` key as the boolean true.)
        loss: ``mse``; ``poisson`` (softplus rate, non-negative count targets);
            ``cosine`` (1 - cosine per row, for direction-only targets).
        hidden: 0 for a linear head, else the width of one GELU hidden layer.
        target_key: batch key of the targets.
    """

    name = "track_head"
    eps = 1e-6

    def __init__(self, layers: Union[str, Sequence[str]] = "transformer_tower",
                 source: str = "delta",
                 loss: str = "mse", hidden: int = 0, target_key: str = "target",
                 weight: float = 1.0, name: Optional[str] = None):
        super().__init__(weight, name)
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        if loss not in TRACK_LOSSES:
            raise ValueError(f"loss must be one of {TRACK_LOSSES}, got {loss!r}")
        self.layers = _as_list(layers)
        self.source = source
        self.loss = loss
        self.hidden = int(hidden)
        self.target_key = target_key
        self.batch_keys = (target_key,)

    def init(self, rng, outputs, batch) -> dict:
        import jax
        import jax.numpy as jnp

        d_in = sum(outputs[self.source][layer].shape[-1] for layer in self.layers)
        d_out = batch[self.target_key].shape[-1]
        widths = [d_in] + ([self.hidden] if self.hidden else []) + [d_out]
        keys = jax.random.split(rng, len(widths) - 1)
        return {f"layer_{i}": {"w": jax.random.normal(k, (a, b)) / jnp.sqrt(a),
                               "b": jnp.zeros((b,))}
                for i, (k, a, b) in enumerate(zip(keys, widths[:-1], widths[1:]))}

    def predict(self, params, outputs):
        """``[B, T]``: the head's output (a rate after softplus for ``poisson``)."""
        import jax
        import jax.numpy as jnp

        x = _source(outputs, self.source, self.layers)
        n = len(params)
        for i in range(n):
            x = x @ params[f"layer_{i}"]["w"] + params[f"layer_{i}"]["b"]
            if i < n - 1:
                x = jax.nn.gelu(x)
        return jax.nn.softplus(x) if self.loss == "poisson" else x.astype(jnp.float32)

    def __call__(self, outputs, batch, params=None):
        import jax.numpy as jnp

        pred = self.predict(params, outputs)
        y = jnp.asarray(batch[self.target_key], jnp.float32)
        if self.loss == "mse":
            return jnp.mean((pred - y) ** 2)
        if self.loss == "poisson":
            return jnp.mean(pred - y * jnp.log(pred + self.eps))
        num = jnp.sum(pred * y, axis=-1)
        den = jnp.linalg.norm(pred, axis=-1) * jnp.linalg.norm(y, axis=-1) + self.eps
        return jnp.mean(1.0 - num / den)


def _pairwise_sq(x):
    """``[B, B]`` squared Euclidean distances, exactly 0 on the diagonal."""
    import jax.numpy as jnp

    diff = x[:, None, :] - x[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


BANDWIDTHS = ("median", "knn")


def diffusion_operator(x, bandwidth_scale: float = 0.25, zero_diagonal: bool = False,
                       bandwidth: str = "median", k: int = 5):
    """Row-normalised Gaussian kernel ``P = D^-1 K`` on the rows of ``x``: the
    diffusion operator of diffusion maps (Coifman and Lafon, 2006).

    ``K_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))``, with ``sigma`` under
    stop-gradient equal to ``bandwidth_scale`` times

    * ``median``: the median off-diagonal pairwise distance (the default);
    * ``knn``: the median over rows of the distance to the ``k``-th nearest
      other row.

    Both make ``P`` invariant to a uniform scaling of ``x``. The defaults
    (diagonal kept, ``bandwidth_scale=0.25``, median) define the operator; with
    ``zero_diagonal`` the self-loops ``K_ii`` are set to 0 before normalising.
    """
    import jax
    import jax.numpy as jnp

    if bandwidth not in BANDWIDTHS:
        raise ValueError(f"bandwidth must be one of {BANDWIDTHS}, got {bandwidth!r}")
    x = jnp.asarray(x, jnp.float32)
    sq = _pairwise_sq(x)
    n = x.shape[0]
    dist = jnp.sqrt(jax.lax.stop_gradient(sq))
    if bandwidth == "median":
        scale = jnp.median(dist[jnp.triu_indices(n, 1)])
    else:
        if not 1 <= int(k) < n:
            raise ValueError(f"knn bandwidth needs 1 <= k < batch size ({n}), got k={k}")
        # column 0 of each sorted row is the row itself (distance 0)
        scale = jnp.median(jnp.sort(dist, axis=1)[:, int(k)])
    sigma = jax.lax.stop_gradient(bandwidth_scale * scale)
    sigma = jnp.maximum(sigma, 1e-12)
    kernel = jnp.exp(-sq / (2 * sigma ** 2))
    if zero_diagonal:
        kernel = kernel * (1.0 - jnp.eye(n, dtype=kernel.dtype))
    return kernel / jnp.sum(kernel, axis=1, keepdims=True)


class DiffusionOperatorMatch(Objective):
    """Match the student's diffusion operator to a teacher's on each minibatch.

    With ``P_S`` and ``P_T`` from :func:`diffusion_operator` (Coifman and
    Lafon, 2006) on the student embeddings (the ``source`` states at
    ``layers``) and the teacher embeddings (``batch[teacher_key]``,
    precomputed, any width), the loss is the mean over rows ``i`` of
    ``(KL(P_T[i] || P_S[i]) + KL(P_S[i] || P_T[i])) / 2``, with ``eps`` added
    inside the logarithms. It is 0 when the operators agree, and invariant to
    rotations and uniform scalings of either embedding. The teacher must not
    be built from labels that evaluate the model (see the module docstring).
    Needs a batch of at least 3 variants.

    The defaults (diagonal kept, ``bandwidth_scale=0.25``, median bandwidth)
    define the objective. For high-dimensional inputs without cluster
    structure, pairwise distances concentrate near their median, so every
    off-diagonal kernel entry is about ``exp(-8)`` and ``P`` is close to the
    identity: the loss is then small for any pair of embeddings and carries
    little signal. On 32 rows of standard normal noise, the mean diagonal of
    ``P`` is 0.92 at width 16 and 0.99 at width 1536, and the loss against an
    unrelated 64-wide teacher falls from 0.16 to 0.012 (0.73 at width 1536
    with ``zero_diagonal``). ``zero_diagonal`` removes the self-loop that
    dominates. ``bandwidth_scale`` moves ``P`` between near-identity (small)
    and near-uniform (large; at 1.0 the same loss is 0.003), and
    ``bandwidth='knn'`` ties ``sigma`` to local neighbourhoods instead. Check
    the diagonal of ``P`` on real batches before relying on the default.

    Args:
        bandwidth_scale, zero_diagonal, bandwidth, k: passed to
            :func:`diffusion_operator`, the same for student and teacher.
    """

    name = "diffusion_operator_match"

    def __init__(self, layers: Union[str, Sequence[str]] = "transformer_tower",
                 source: str = "delta",
                 teacher_key: str = "teacher", bandwidth_scale: float = 0.25,
                 zero_diagonal: bool = False, bandwidth: str = "median", k: int = 5,
                 eps: float = 1e-8, weight: float = 1.0, name: Optional[str] = None):
        super().__init__(weight, name)
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        if bandwidth not in BANDWIDTHS:
            raise ValueError(f"bandwidth must be one of {BANDWIDTHS}, got {bandwidth!r}")
        self.layers = _as_list(layers)
        self.source = source
        self.teacher_key = teacher_key
        self.bandwidth_scale = float(bandwidth_scale)
        self.zero_diagonal = bool(zero_diagonal)
        self.bandwidth = bandwidth
        self.k = int(k)
        self.eps = float(eps)
        self.batch_keys = (teacher_key,)

    def operator(self, x):
        return diffusion_operator(x, self.bandwidth_scale, self.zero_diagonal, self.bandwidth,
                                  self.k)

    def loss(self, student, teacher):
        import jax.numpy as jnp

        p_s = self.operator(student)
        p_t = self.operator(teacher)
        log_s, log_t = jnp.log(p_s + self.eps), jnp.log(p_t + self.eps)
        kl_ts = jnp.sum(p_t * (log_t - log_s), axis=1)
        kl_st = jnp.sum(p_s * (log_s - log_t), axis=1)
        return jnp.mean((kl_ts + kl_st) / 2)

    def __call__(self, outputs, batch, params=None):
        return self.loss(_source(outputs, self.source, self.layers), batch[self.teacher_key])


def jacobian_frobenius(fn, x, rng=None, probes: int = 1, mode: str = "hutchinson"):
    """Mean over rows of ``x`` of ``||J_fn(x_i)||_F^2``; ``fn`` maps ``[D] -> [D']``.

    ``exact`` builds each Jacobian with ``jax.jacfwd``. ``hutchinson``
    estimates ``E_v ||J v||^2`` with ``probes`` Rademacher vectors per row,
    each ``J v`` from one ``jax.jvp``: unbiased, and no Jacobian is formed.
    """
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(x, jnp.float32)
    if mode == "exact":
        jac = jax.vmap(jax.jacfwd(fn))(x)
        return jnp.mean(jnp.sum(jac ** 2, axis=(-2, -1)))
    if mode != "hutchinson":
        raise ValueError(f"mode must be 'exact' or 'hutchinson', got {mode!r}")
    if rng is None:
        raise ValueError("hutchinson mode needs rng")
    v = jax.random.rademacher(rng, (probes,) + x.shape, dtype=jnp.float32)

    def jv(xi, vi):
        return jax.jvp(fn, (xi,), (vi,))[1]

    jvs = jax.vmap(lambda vp: jax.vmap(jv)(x, vp))(v)
    return jnp.mean(jnp.sum(jvs ** 2, axis=-1))


class JacobianPenalty(Objective):
    """``||J||_F^2`` of the edit at the edit layer ("Measure before you look",
    UniReps 2025): the Jacobian of the edited site vector with respect to the
    unedited one, at ``outputs['edit_input']``.

    Args:
        mode: ``hutchinson`` (``probes`` random probes through ``jax.jvp``) or
            ``exact`` (``jax.jacfwd``; for tests and small widths).
        residual: penalise the Jacobian of the edit alone, ``h' - h``, instead
            of ``h'``: 0 at initialisation, and free of the constant ``d``
            that ``||I||_F^2`` adds.

    Only the edit's own map is differentiated, not AlphaGenome. With
    :class:`~.interventions.HeadOnly` it is a constant.
    """

    name = "jacobian_penalty"

    def __init__(self, mode: str = "hutchinson", probes: int = 1, residual: bool = False,
                 weight: float = 1.0, name: Optional[str] = None):
        super().__init__(weight, name)
        if mode not in ("hutchinson", "exact"):
            raise ValueError(f"mode must be 'exact' or 'hutchinson', got {mode!r}")
        self.mode = mode
        self.probes = int(probes)
        self.residual = bool(residual)

    def __call__(self, outputs, batch, params=None):
        import jax

        fn = outputs["edit_fn"]
        if self.residual:
            edit = fn
            fn = lambda h: edit(h) - h  # noqa: E731
        rng = jax.random.fold_in(outputs["rng"], 1) if "rng" in outputs else None
        return jacobian_frobenius(fn, outputs["edit_input"], rng, self.probes, self.mode)


class CovarianceIsotropy(Objective):
    """Off-diagonal share of the batch covariance of the states:
    ``||C - diag(C)||_F^2 / ||C||_F^2`` in [0, 1); 0 for uncorrelated dimensions.
    Scale-free; the batch must be larger than 1."""

    name = "covariance_isotropy"

    def __init__(self, layers: Union[str, Sequence[str]] = "transformer_tower",
                 source: str = "delta",
                 weight: float = 1.0, name: Optional[str] = None):
        super().__init__(weight, name)
        self.layers = _as_list(layers)
        self.source = source

    def __call__(self, outputs, batch, params=None):
        import jax.numpy as jnp

        x = _source(outputs, self.source, self.layers)
        x = x - jnp.mean(x, axis=0)
        c = x.T @ x / (x.shape[0] - 1)
        off = c - jnp.diag(jnp.diag(c))
        return jnp.sum(off ** 2) / (jnp.sum(c ** 2) + 1e-12)


class Composite:
    """Weighted sum of objectives; their parameters are kept under their names.

    ``terms`` returns each objective's unweighted value, for logging.
    """

    def __init__(self, objectives: Sequence[Objective]):
        objectives = list(objectives)
        names = [o.name for o in objectives]
        if not objectives or len(set(names)) != len(names):
            raise ValueError(f"objectives must be non-empty with distinct names, got {names}; "
                             "pass name= to tell two of one kind apart")
        self.objectives = objectives
        self.name = "+".join(names)

    @property
    def batch_keys(self) -> tuple:
        return tuple(k for o in self.objectives for k in o.batch_keys)

    def find(self, kind) -> Optional[Objective]:
        """The first objective that is an instance of ``kind``, or None."""
        return next((o for o in self.objectives if isinstance(o, kind)), None)

    def init(self, rng, outputs, batch) -> dict:
        import jax

        keys = jax.random.split(rng, len(self.objectives))
        return {o.name: o.init(k, outputs, batch) for o, k in zip(self.objectives, keys)}

    def terms(self, outputs, batch, params=None) -> dict:
        params = params or {}
        return {o.name: o(outputs, batch, params.get(o.name, {})) for o in self.objectives}

    def __call__(self, outputs, batch, params=None):
        values = self.terms(outputs, batch, params)
        return sum(o.weight * values[o.name] for o in self.objectives)


def as_composite(objective) -> Composite:
    """``objective`` as a :class:`Composite` (a list of objectives, one, or a Composite)."""
    if isinstance(objective, Composite):
        return objective
    if isinstance(objective, (list, tuple)):
        return Composite(objective)
    return Composite([objective])
