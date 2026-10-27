"""AlphaGenome site states as a LatentModule: one row per variant.

``transform`` returns, for the datamodule's variants, either the alt - ref
delta at one layer or the scalar features (delta_norm, cosine, pr3) at several
layers. With that, ``manylatents.main`` runs its metrics and plots on any layer
without new code. The states come from a local model, or from the output of
``python -m manylatents.dogma.alphagenome.site_states`` (``states_dir``), which
is how 1 Mb runs on a cluster feed a later analysis.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.dogma.alphagenome.taps import LAYERS
from manylatents.dogma.latent_features import cosine, delta_norm, pr3

OUTPUTS = ("delta", "features")


def datamodule_variants(module):
    """chrom, pos, ref, alt (with ``chromosome`` and ``variant_id``) from
    ``module.datamodule.get_variants()``; no other column is read."""
    name = type(module).__name__
    if module.datamodule is None:
        raise ValueError(f"{name} requires datamodule with get_variants(). "
                         "Pass datamodule= when instantiating the encoder.")
    if not hasattr(module.datamodule, "get_variants"):
        raise ValueError(f"Datamodule {type(module.datamodule).__name__} has no get_variants() "
                         "method returning chrom, pos, ref, alt.")
    from manylatents.dogma.variants import VARIANT_COLUMNS, with_variant_ids

    table = module.datamodule.get_variants()
    return with_variant_ids(table[[c for c in VARIANT_COLUMNS if c in table.columns]])


def row_indices(x, n_rows: int) -> np.ndarray:
    """``x`` (``[N, 1]`` or ``[N]``, numpy or torch) as int64 row indices.

    Raises:
        ValueError: ``x`` is not one column of whole numbers in ``[0, n_rows)``.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError(
            f"x must be a column of row indices into the datamodule's get_variants() "
            f"(shape [N, 1]), got shape {x.shape}; use a datamodule whose datasets yield "
            "row indices, such as VariantTableDataModule")
    if not np.issubdtype(x.dtype, np.number) or (
            len(x) and not np.all(np.isfinite(x) & (x == np.round(x)))):
        raise ValueError("x must hold whole-number row indices into get_variants()")
    index = x.astype(np.int64)
    if len(index) and (index.min() < 0 or index.max() >= n_rows):
        raise ValueError(f"row indices of x must be in [0, {n_rows}), got "
                         f"[{index.min()}, {index.max()}]")
    return index


def variants_at(module, x):
    """``(variants, index)``: the rows of ``get_variants()`` that ``x`` indexes,
    in the order of ``x`` (all rows, in order, when ``x`` is None)."""
    variants = datamodule_variants(module)
    if x is None:
        return variants, np.arange(len(variants))
    index = row_indices(x, len(variants))
    return variants.iloc[index].reset_index(drop=True), index


class AlphaGenomeSiteEncoder(LatentModule):
    """Per-variant AlphaGenome hidden-state responses at the variant's site.

    Args:
        layer: layer for ``output='delta'`` (a name in ``taps.LAYERS``).
        output: ``delta`` -> ``(N, D)`` alt - ref at ``layer``; ``features`` ->
            ``(N, 3 * len(feature_layers))``, columns delta_norm, cosine, pr3
            for each layer in turn (see ``feature_names``).
        feature_layers: layers for ``output='features'``; all 26 if None.
        fasta: reference genome; required unless ``states_dir`` is given.
        states_dir: read precomputed site_states output instead of running
            the model. Its run must hold the needed layers (and, for pr3,
            ``--alleles all``).
        sequence_length, model_version, organism, hf_cache, window, batch_size:
            as for the site_states command.
        repeat: which repeat to read from ``states_dir``.
        policy: compute dtype of the model, ``bfloat16`` (default) or
            ``float32`` (see ``model.policy_string``).

    The datamodule must provide ``get_variants()``: a table with chrom, pos,
    ref, alt. ``x`` holds row indices into that table (``[N, 1]``, as the
    datasets of :class:`~manylatents.dogma.data.VariantTableDataModule`
    yield); ``transform(x)`` returns one row per index, in the order of
    ``x``, and ``transform(None)`` one row per variant of the table. A variant
    that cannot be scored (not an SNV, reference mismatch, window off the
    chromosome) raises rather than being dropped, so rows stay aligned with
    the datamodule's labels.
    """

    def __init__(self, layer: str = "transformer_tower", output: str = "delta",
                 feature_layers: Optional[Sequence[str]] = None, fasta: Optional[str] = None,
                 states_dir: Optional[str] = None, sequence_length: int = 1_048_576,
                 model_version: str = "fold_0", organism: str = "HOMO_SAPIENS",
                 hf_cache: Optional[str] = None, window: int = 0, batch_size: int = 1,
                 repeat: int = 0, policy: str = "bfloat16", **kwargs):
        kwargs.setdefault("n_components", 0)
        super().__init__(**kwargs)
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}; known: {list(LAYERS)}")
        feature_layers = list(LAYERS) if feature_layers is None else list(feature_layers)
        unknown = [name for name in feature_layers if name not in LAYERS]
        if unknown or not feature_layers:
            raise ValueError(f"feature_layers must be non-empty names in taps.LAYERS; "
                             f"unknown: {unknown}")
        if fasta is None and states_dir is None:
            raise ValueError("pass fasta (to run the model) or states_dir (precomputed states)")
        self.layer = layer
        self.output = output
        self.feature_layers = feature_layers
        self.fasta = fasta
        self.states_dir = states_dir
        self.sequence_length = int(sequence_length)
        self.model_version = model_version
        self.organism = organism
        self.hf_cache = hf_cache
        self.window = int(window)
        self.batch_size = int(batch_size)
        self.repeat = int(repeat)
        from manylatents.dogma.alphagenome.model import policy_string

        policy_string(policy)
        self.policy = policy
        self._model = None

    @property
    def layers(self) -> list[str]:
        return [self.layer] if self.output == "delta" else list(self.feature_layers)

    @property
    def feature_names(self) -> list[str]:
        """Column names of the result for ``output='features'``."""
        return [f"{layer}/{name}" for layer in self.feature_layers
                for name in ("delta_norm", "cosine", "pr3")]

    def _load_model(self):
        if self._model is None:
            from manylatents.dogma.alphagenome.model import load_local

            self._model = load_local(model_version=self.model_version, organism=self.organism,
                                     hf_cache=self.hf_cache, policy=self.policy)
            self._model.check_length(self.sequence_length)
        return self._model

    def fit(self, x=None, y=None) -> None:
        """Nothing to fit: the model is pretrained."""
        self._is_fitted = True

    def _from_states_dir(self, variants) -> np.ndarray:
        from manylatents.dogma.alphagenome.site_states import delta_matrix, read_site_table

        ids = variants["variant_id"].tolist()
        if self.output == "delta":
            return delta_matrix(self.states_dir, self.layer, ids, repeat=self.repeat)
        table = read_site_table(self.states_dir, "features")
        table = table[table["repeat"] == self.repeat].set_index(["variant_id", "layer"])
        columns = []
        for layer in self.feature_layers:
            index = [(vid, layer) for vid in ids]
            absent = [key for key in index if key not in table.index]
            if absent:
                raise KeyError(f"{len(absent)} variants have no {layer!r} features beneath "
                               f"{self.states_dir} (first: {absent[:5]})")
            columns.append(table.loc[index, ["delta_norm", "cosine", "pr3"]].to_numpy(np.float32))
        return np.concatenate(columns, axis=1)

    def _from_model(self, variants) -> np.ndarray:
        import pyfaidx

        from manylatents.dogma.alphagenome.site_states import prepare_variants, score_variants

        model = self._load_model()
        alleles = "ref_alt" if self.output == "delta" else "all"
        with pyfaidx.Fasta(self.fasta, sequence_always_upper=True) as fasta:
            kept, skipped = prepare_variants(variants, fasta, self.sequence_length)
            if len(skipped):
                raise ValueError(
                    f"{len(skipped)} variants cannot be scored "
                    f"{skipped['reason'].value_counts().to_dict()} (first: "
                    f"{skipped['variant_id'].tolist()[:5]}); drop them from the datamodule")
            states, failed = score_variants(
                model, kept, fasta, self.sequence_length, self.layers, alleles=alleles,
                window=self.window, batch_size=self.batch_size)
        if failed:
            raise RuntimeError(f"{len(failed)} variants ran out of device memory: {failed[:3]}")
        rows = []
        for vid in variants["variant_id"]:
            h = states[vid]
            if self.output == "delta":
                rows.append(h[("alt", 0)][self.layer] - h[("ref", 0)][self.layer])
                continue
            row = []
            for layer in self.feature_layers:
                ref, alt = h[("ref", 0)][layer], h[("alt", 0)][layer]
                row += [delta_norm(ref, alt), cosine(ref, alt),
                        pr3(ref, alt, h[("alt2", 0)][layer], h[("alt3", 0)][layer])]
            rows.append(np.asarray(row))
        return np.stack(rows).astype(np.float32)

    def transform(self, x=None) -> np.ndarray:
        """``(N, D)`` float32, one row per row index in ``x`` (see the class docstring)."""
        variants, _ = variants_at(self, x)
        if variants.empty:
            return np.zeros((0, 0), dtype=np.float32)
        if self.states_dir is not None:
            return self._from_states_dir(variants)
        return self._from_model(variants)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(output={self.output!r}, layers={self.layers!r}, "
                f"sequence_length={self.sequence_length})")
