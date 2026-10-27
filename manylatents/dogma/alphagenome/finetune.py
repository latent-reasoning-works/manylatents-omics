"""Light finetuning or steering of AlphaGenome's representations, as a LatentModule.

``fit`` trains an intervention (:mod:`.interventions`) and the objectives'
heads (:mod:`.objectives`) on the variants that ``x`` indexes with a
:class:`~.train.JaxTrainer`, every AlphaGenome parameter frozen; ``transform``
returns one row per index of its ``x`` from the edited model. ``x`` holds row
indices into the datamodule's ``get_variants()`` (as the datasets of
:class:`~manylatents.dogma.data.VariantTableDataModule` yield), so ``fit``
sees only the training rows. The JAX trainer is a nested
Hydra object of this module's config: ``manylatents.main`` builds its own
Lightning trainer, which a LatentModule does not use.

Targets never come from ``fit``'s ``y`` unless ``targets_from='y'``. ``y`` is
the datamodule's labels; a label that also evaluates the model should not
train it, or the evaluation is circular. Supply targets (MPRA or caQTL
effects, AlphaGenome track deltas) through ``targets_path``, or use
``targets_from='y'`` with labels that no evaluation reads.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.dogma.alphagenome.encoder import variants_at
from manylatents.dogma.alphagenome.taps import LAYERS

logger = logging.getLogger(__name__)

TRANSFORM_OUTPUTS = ("edited_site_state", "delta", "head_output")
TARGET_SOURCES = ("path", "y")
_ID_COLUMNS = ("chrom", "pos", "ref", "alt", "chromosome", "variant_id")

_CODES = np.full(256, 4, dtype=np.uint8)
for _i, _base in enumerate("ACGT"):
    _CODES[ord(_base)] = _CODES[ord(_base.lower())] = _i


def encode(sequence: str) -> np.ndarray:
    """``[len]`` uint8 base codes, A C G T = 0..3 and anything else (N) = 4.
    ``jax.nn.one_hot(codes, 4)`` gives :func:`~.model.one_hot`'s encoding."""
    return _CODES[np.frombuffer(sequence.encode("latin1"), dtype=np.uint8)]


def _instantiate(obj):
    """Hydra configs that arrive un-instantiated (``_recursive_: false``) are built here."""
    if isinstance(obj, dict) or type(obj).__name__ == "DictConfig":
        if "_target_" in obj:
            from hydra.utils import instantiate

            return instantiate(obj)
    if isinstance(obj, (list, tuple)) or type(obj).__name__ == "ListConfig":
        return [_instantiate(o) for o in obj]
    return obj


def load_rows(path, variants, what: str = "targets", rows=None,
              n_rows: Optional[int] = None) -> np.ndarray:
    """``[N, T]`` float32 rows for ``variants`` (in their order) from ``path``.

    ``.npy``: one row per variant of the datamodule's ``get_variants()``, in
    its order; ``rows`` (positions in that table, ``n_rows`` long) selects
    those of ``variants``. ``.npz``: arrays ``variant_id`` and ``values``.
    ``.parquet``/``.tsv``/``.csv``: a ``variant_id`` column, or chrom, pos,
    ref and alt; every other numeric column is a value.

    Raises:
        KeyError: a variant has no row.
        ValueError: an ``.npy`` file with a different number of rows.
    """
    import pandas as pd

    path = Path(path)
    n = len(variants)
    if path.suffix == ".npy":
        values = np.load(path)
        expected = n if rows is None else n_rows
        if len(values) != expected:
            raise ValueError(f"{what} {path} has {len(values)} rows for {expected} variants")
        values = np.asarray(values, np.float32).reshape(len(values), -1)
        return values if rows is None else values[np.asarray(rows)]
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            table = pd.DataFrame(np.asarray(data["values"], np.float32).reshape(
                len(data["variant_id"]), -1))
            table.index = data["variant_id"].astype(str)
    else:
        if path.suffix == ".parquet":
            table = pd.read_parquet(path)
        else:
            table = pd.read_csv(path, sep="," if path.suffix == ".csv" else "\t")
        if "variant_id" not in table.columns:
            from manylatents.dogma.variants import with_variant_ids

            table = with_variant_ids(table)
        table = table.set_index("variant_id")
        table = table[[c for c in table.columns
                       if c not in _ID_COLUMNS and pd.api.types.is_numeric_dtype(table[c])]]
    ids = variants["variant_id"].tolist()
    absent = [vid for vid in ids if vid not in table.index]
    if absent:
        raise KeyError(f"{len(absent)} variants have no {what} in {path} (first: {absent[:5]})")
    return table.loc[ids].to_numpy(np.float32)


class AlphaGenomeFinetune(LatentModule):
    """Train an edit of AlphaGenome's states (or heads on frozen states) and
    return per-variant representations.

    Args:
        intervention: :class:`~.interventions.LowRankEdit` (default, rank 8)
            or :class:`~.interventions.HeadOnly`, or its Hydra config.
        objective: an objective, a list of them, or a
            :class:`~.objectives.Composite` (or Hydra configs of these).
        trainer: a :class:`~.train.JaxTrainer` or its Hydra config.
        layer: tap where the edit is applied (a name in ``taps.LAYERS``);
            ignored by HeadOnly.
        readout_layers: layers whose site states the objectives and
            ``transform`` read, concatenated in this order; ``[layer]`` if None.
            A readout after ``layer`` backpropagates through the frozen layers
            in between.
        fasta: reference genome.
        sequence_length: input length. 131,072 is meant to let gradients
            through the tower fit on one H100; 1,048,576 is allowed (see docs).
            Without HeadOnly, every variant's reference window is kept in host
            memory as uint8 codes, ``sequence_length`` bytes per variant: 1 MiB
            per variant at 1 Mb (10 GiB for 10,000 variants), 128 KiB at
            131,072.
        model_version, organism, hf_cache: as for ``load_local``.
        window: bins on each side of the site's bin to average (and, with
            ``positions='site'``, to edit).
        targets_from: ``path`` (``targets_path``) or ``y`` (``fit``'s labels;
            only for data that no evaluation uses, see the module docstring).
        targets_path, teacher_path: per-variant rows for ``batch['target']``
            and ``batch['teacher']`` (see :func:`load_rows`).
        transform_output: ``edited_site_state`` (alt allele's edited state),
            ``delta`` (edited alt - ref) or ``head_output`` (the
            :class:`~.objectives.TrackHead`'s prediction).
        val_fraction: share of variants held out to evaluate the objective
            (the trainer's ``eval_every`` and early stop); 0 for none.
        inference_batch_size: variants per forward without gradients (the
            HeadOnly state cache, ``transform``); each variant is two sequences.
        model: a preloaded ``LocalModel``; loaded from ``hf_cache`` if None.
        policy: compute dtype when the model is loaded here, ``bfloat16``
            (default) or ``float32`` (see ``model.policy_string``).

    ``fit(x, y)`` and ``transform(x)`` read ``x`` as row indices into the
    datamodule's ``get_variants()`` (``None``: every row) and raise
    ValueError for anything else; rows of ``transform`` follow ``x``. With
    ``targets_from='y'``, row ``i`` of ``y`` is the target of index ``i`` of
    ``x``. A variant that cannot be scored raises. With HeadOnly the site states are
    computed once and cached; otherwise every step runs the model on the
    minibatch's ref and alt sequences.
    """

    def __init__(self, intervention: Any = None, objective: Any = None, trainer: Any = None,
                 layer: str = "mha_block_4_input", readout_layers: Optional[Sequence[str]] = None,
                 fasta: Optional[str] = None, sequence_length: int = 131_072,
                 model_version: str = "fold_0", organism: str = "HOMO_SAPIENS",
                 hf_cache: Optional[str] = None, window: int = 0, targets_from: str = "path",
                 targets_path: Optional[str] = None, teacher_path: Optional[str] = None,
                 transform_output: str = "edited_site_state", val_fraction: float = 0.0,
                 inference_batch_size: int = 2, model: Any = None, policy: str = "bfloat16",
                 **kwargs):
        kwargs.setdefault("n_components", 0)
        super().__init__(**kwargs)
        from manylatents.dogma.alphagenome.interventions import HeadOnly, LowRankEdit
        from manylatents.dogma.alphagenome.objectives import TrackHead, as_composite
        from manylatents.dogma.alphagenome.train import JaxTrainer

        self.intervention = _instantiate(intervention) if intervention is not None \
            else LowRankEdit()
        if objective is None:
            raise ValueError("objective is required (e.g. objectives.TrackHead)")
        self.objective = as_composite(_instantiate(objective))
        self.trainer = _instantiate(trainer) if trainer is not None else JaxTrainer()
        self.head_only = isinstance(self.intervention, HeadOnly)
        readout_layers = [layer] if readout_layers is None else list(readout_layers)
        unknown = [name for name in [layer, *readout_layers] if name not in LAYERS]
        if unknown or not readout_layers:
            raise ValueError(f"layer and readout_layers must be names in taps.LAYERS; "
                             f"unknown: {unknown}")
        for o in self.objective.objectives:
            missing = [name for name in getattr(o, "layers", ()) if name not in readout_layers]
            if missing:
                raise ValueError(f"objective {o.name!r} reads {missing}, which are not in "
                                 f"readout_layers {readout_layers}")
        if self.head_only and any(type(o).__name__ == "JacobianPenalty"
                                  for o in self.objective.objectives):
            raise ValueError("JacobianPenalty needs an edit; HeadOnly has none")
        if targets_from not in TARGET_SOURCES:
            raise ValueError(f"targets_from must be one of {TARGET_SOURCES}, got {targets_from!r}")
        if transform_output not in TRANSFORM_OUTPUTS:
            raise ValueError(f"transform_output must be one of {TRANSFORM_OUTPUTS}, "
                             f"got {transform_output!r}")
        self.track_head = self.objective.find(TrackHead)
        if transform_output == "head_output" and self.track_head is None:
            raise ValueError("transform_output='head_output' needs a TrackHead objective")
        if not 0 <= float(val_fraction) < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if fasta is None:
            raise ValueError("fasta is required: the model reads the reference sequence")
        self.layer = layer
        self.readout_layers = readout_layers
        self.fasta = fasta
        self.sequence_length = int(sequence_length)
        self.model_version = model_version
        self.organism = organism
        self.hf_cache = hf_cache
        self.window = int(window)
        self.targets_from = targets_from
        self.targets_path = targets_path
        self.teacher_path = teacher_path
        self.transform_output = transform_output
        self.val_fraction = float(val_fraction)
        self.inference_batch_size = int(inference_batch_size)
        from manylatents.dogma.alphagenome.model import policy_string

        policy_string(policy)
        self.policy = policy
        self._model = model
        self._cache: dict = {}
        self.params = None
        self.history: list = []

    # -- model and data ---------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from manylatents.dogma.alphagenome.model import load_local

            self._model = load_local(model_version=self.model_version, organism=self.organism,
                                     hf_cache=self.hf_cache, policy=self.policy)
        self._model.check_length(self.sequence_length)
        return self._model

    @property
    def site_index(self) -> int:
        from manylatents.dogma.alphagenome.model import site_window

        return site_window(self.sequence_length, 1, self.sequence_length)[1]

    def _inputs(self, variants) -> dict:
        """Per-variant model inputs: cached site states for HeadOnly
        (``ref/<layer>``, ``alt/<layer>``), else base codes of the reference
        window (``codes``, ``[N, L]`` uint8) and the alt base (``alt_code``)."""
        key = tuple(variants["variant_id"])
        if key in self._cache:
            return self._cache[key]
        import pyfaidx

        from manylatents.dogma.alphagenome.model import site_sequences
        from manylatents.dogma.alphagenome.site_states import prepare_variants, score_variants

        model = self._load_model()
        with pyfaidx.Fasta(self.fasta, sequence_always_upper=True) as fasta:
            kept, skipped = prepare_variants(variants, fasta, self.sequence_length)
            if len(skipped):
                raise ValueError(
                    f"{len(skipped)} variants cannot be scored "
                    f"{skipped['reason'].value_counts().to_dict()} (first: "
                    f"{skipped['variant_id'].tolist()[:5]}); drop them from the datamodule")
            if self.head_only:
                states, failed = score_variants(
                    model, kept, fasta, self.sequence_length, self.readout_layers,
                    alleles="ref_alt", window=self.window,
                    batch_size=2 * self.inference_batch_size)
                if failed:
                    raise RuntimeError(f"{len(failed)} variants ran out of device memory: "
                                       f"{failed[:3]}")
                inputs = {f"{allele}/{layer}": np.stack(
                    [states[vid][(allele, 0)][layer] for vid in key]).astype(np.float32)
                    for allele in ("ref", "alt") for layer in self.readout_layers}
            else:
                codes = np.empty((len(kept), self.sequence_length), dtype=np.uint8)
                alt_code = np.empty(len(kept), dtype=np.int32)
                for i, row in enumerate(kept.itertuples(index=False)):
                    sites = site_sequences(row.chromosome, row.pos, row.ref, row.alt, fasta,
                                           self.sequence_length, alleles="ref_alt")
                    assert sites.site_index == self.site_index
                    codes[i] = encode(sites.sequences["ref"])
                    alt_code[i] = encode(sites.bases["alt"])[0]
                inputs = {"codes": codes, "alt_code": alt_code}
        self._cache = {key: inputs}  # one set of variants at a time
        return inputs

    def _supervision(self, variants, y, rows=None, n_rows=None) -> dict:
        """``target`` and ``teacher`` rows the objectives need."""
        needed = set(self.objective.batch_keys)
        unknown = needed - {"target", "teacher"}
        if unknown:
            raise ValueError(f"objectives read batch keys {sorted(unknown)}; this module "
                             "supplies only 'target' and 'teacher'")
        out = {}
        if "target" in needed:
            if self.targets_from == "y":
                if y is None:
                    raise ValueError("targets_from='y' but fit() got no y")
                logger.warning("training targets come from fit()'s y; these must not be "
                               "labels that any evaluation of this model uses")
                y = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
                if len(y) != len(variants):
                    raise ValueError(f"y has {len(y)} rows for {len(variants)} rows of x")
                out["target"] = np.asarray(y, np.float32).reshape(len(variants), -1)
            elif self.targets_path is None:
                raise ValueError(
                    "the objective needs targets: set targets_path. fit()'s y is not used "
                    "unless targets_from='y', because benchmark labels must not train the "
                    "model they evaluate")
            else:
                out["target"] = load_rows(self.targets_path, variants, "targets", rows, n_rows)
        if "teacher" in needed:
            if self.teacher_path is None:
                raise ValueError("the objective needs a teacher: set teacher_path")
            out["teacher"] = load_rows(self.teacher_path, variants, "teacher embeddings", rows,
                                       n_rows)
        return out

    # -- forward ----------------------------------------------------------------------------

    @functools.cached_property
    def _forward(self):
        from manylatents.dogma.alphagenome.interventions import make_forward

        return make_forward(self._load_model(), self.intervention,
                            None if self.head_only else self.layer, self.readout_layers,
                            self.site_index, window=self.window, remat=self.trainer.remat)

    def _outputs(self, trainable, frozen, batch, rng=None) -> dict:
        """The ``outputs`` dict of :mod:`.objectives` for one minibatch."""
        import jax
        import jax.numpy as jnp

        if self.head_only:
            ref = {layer: batch[f"ref/{layer}"] for layer in self.readout_layers}
            alt = {layer: batch[f"alt/{layer}"] for layer in self.readout_layers}
            edit_input = None
        else:
            ref_onehot = jax.nn.one_hot(batch["codes"], 4, dtype=jnp.float32)
            alt_onehot = ref_onehot.at[:, self.site_index].set(
                jax.nn.one_hot(batch["alt_code"], 4, dtype=jnp.float32))
            onehot = jnp.concatenate([ref_onehot, alt_onehot])
            organism = jnp.full((onehot.shape[0],), self._load_model().organism_index, jnp.int32)
            states, edit_input = self._forward(trainable["intervention"], frozen["params"],
                                               frozen["state"], onehot, organism)
            b = ref_onehot.shape[0]
            ref = {k: v[:b] for k, v in states.items()}
            alt = {k: v[b:] for k, v in states.items()}
        edit_params = trainable["intervention"]
        outputs = {"ref": ref, "alt": alt,
                   "delta": {k: alt[k] - ref[k] for k in ref},
                   "edit_fn": lambda h: self.intervention.apply(edit_params, h),
                   "edit": self.intervention, "edit_params": edit_params}
        if edit_input is not None:
            outputs["edit_input"] = edit_input
        if rng is not None:
            outputs["rng"] = rng
        return outputs

    def _loss_fn(self, trainable, frozen, batch, rng):
        outputs = self._outputs(trainable, frozen, batch, rng)
        terms = self.objective.terms(outputs, batch, trainable["objective"])
        loss = sum(o.weight * terms[o.name] for o in self.objective.objectives)
        return loss, terms

    def _frozen(self) -> dict:
        if self.head_only:
            return {}
        model = self._load_model()
        return {"params": model.params, "state": model.state}

    # -- LatentModule -----------------------------------------------------------------------

    def _init_params(self, inputs: dict, supervision: dict) -> dict:
        import jax

        from manylatents.dogma.alphagenome.interventions import make_forward

        key = jax.random.PRNGKey(self.init_seed)
        k_edit, k_obj = jax.random.split(key)
        if self.head_only:
            widths = {layer: inputs[f"ref/{layer}"].shape[-1] for layer in self.readout_layers}
            edit_params = {}
        else:
            from manylatents.dogma.alphagenome.interventions import HeadOnly

            model = self._load_model()
            probe = make_forward(model, HeadOnly(), None,
                                 list(dict.fromkeys([self.layer, *self.readout_layers])),
                                 self.site_index, window=self.window)
            shapes, _ = jax.eval_shape(
                probe, {}, model.params, model.state,
                jax.ShapeDtypeStruct((1, self.sequence_length, 4), np.float32),
                jax.ShapeDtypeStruct((1,), np.int32))
            widths = {layer: shapes[layer].shape[-1] for layer in shapes}
            edit_params = self.intervention.init(k_edit, widths[self.layer])
        state = {layer: jax.ShapeDtypeStruct((1, widths[layer]), np.float32)
                 for layer in self.readout_layers}
        shapes = {"ref": state, "alt": state, "delta": state}
        sample = {k: v[:1] for k, v in supervision.items()}
        return {"intervention": edit_params,
                "objective": self.objective.init(k_obj, shapes, sample)}

    def _callback(self):
        if self.trainer.callback is not None:
            return self.trainer.callback
        try:
            import wandb
        except ImportError:
            return None
        if wandb.run is None:
            return None
        return lambda record: wandb.log({f"finetune/{k}": v for k, v in record.items()})

    def fit(self, x=None, y=None) -> None:
        """Train on the variants that ``x`` indexes (see the class docstring)."""
        variants, rows = variants_at(self, x)
        if variants.empty:
            raise ValueError("no variants to train on")
        supervision = self._supervision(variants, y, rows, len(self.datamodule.get_variants()))
        data = {**self._inputs(variants), **supervision}
        trainable = self._init_params(data, supervision)
        eval_data = None
        if self.val_fraction:
            order = np.random.default_rng(self.trainer.seed).permutation(len(variants))
            n_val = max(1, int(round(self.val_fraction * len(variants))))
            val, train = order[:n_val], order[n_val:]
            eval_data = {k: v[val] for k, v in data.items()}
            data = {k: v[train] for k, v in data.items()}
        callback, self.trainer.callback = self.trainer.callback, self._callback()
        try:
            self.params, self.history = self.trainer.fit(self._loss_fn, trainable,
                                                         self._frozen(), data, eval_data)
        finally:
            self.trainer.callback = callback
        self._is_fitted = True

    @functools.cached_property
    def _embed(self):
        import jax
        import jax.numpy as jnp

        def embed(trainable, frozen, batch):
            outputs = self._outputs(trainable, frozen, batch)
            if self.transform_output == "head_output":
                head = self.track_head
                return head.predict(trainable["objective"][head.name], outputs)
            source = "alt" if self.transform_output == "edited_site_state" else "delta"
            return jnp.concatenate([outputs[source][k] for k in self.readout_layers], axis=-1)

        return jax.jit(embed)

    def transform(self, x=None) -> np.ndarray:
        """``(N, d)`` float32, one row per row index in ``x`` (see ``transform_output``)."""
        if self.params is None:
            raise RuntimeError("call fit() before transform()")
        variants, _ = variants_at(self, x)
        if variants.empty:
            return np.zeros((0, 0), dtype=np.float32)
        inputs = self._inputs(variants)
        frozen = self._frozen()
        n = len(variants)
        bs = min(self.inference_batch_size, n)
        rows = []
        for begin in range(0, n, bs):
            index = np.minimum(np.arange(begin, begin + bs), n - 1)
            out = self._embed(self.params, frozen, {k: v[index] for k, v in inputs.items()})
            rows.append(np.asarray(out, np.float32)[:min(bs, n - begin)])
        return np.concatenate(rows)

    def save(self, path) -> Path:
        """Trainable parameters (edit and heads) to ``.npz``."""
        from manylatents.dogma.alphagenome.train import save_params

        if self.params is None:
            raise RuntimeError("nothing to save before fit()")
        return save_params(path, self.params)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(intervention={self.intervention!r}, "
                f"objective={self.objective.name!r}, layer={self.layer!r}, "
                f"readout_layers={self.readout_layers!r}, "
                f"sequence_length={self.sequence_length})")
