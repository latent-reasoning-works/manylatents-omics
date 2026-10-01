"""AlphaGenome Atlas variant scores as variant-by-track matrices.

Atlas serves precomputed AlphaGenome variant scores: for each scorer (roughly,
one assay modality) a matrix with one column per track. This module fetches
them in chunks into a resumable on-disk cache and reads them back aligned to a
caller-given variant order.

Three properties of the client shape the code (see ``alphagenome.atlas.atlas``):
rows come back in completion order, so they are matched through the variant
they carry; one unservable variant fails a whole batch, so a failed batch is
retried one variant at a time; gene-scoped scorers return several rows per
variant, so reading one row per variant requires a named reduction.

Nothing here goes through ``SignalRecord``: one Python object per (variant,
track) does not scale to this data, and that schema requires activity
percentiles Atlas does not provide.

Atlas outputs are subject to Google DeepMind's terms of use. This module only
fetches and reshapes; it fits nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
from numbers import Integral
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from manylatents.dogma.variants import with_variant_ids

API_KEY_ENV = "ALPHAGENOME_API_KEY"
_MANIFEST = "manifest.json"


class AtlasScoresMissing(ValueError):
    """Requested variants have no cached scores for a scorer."""

    def __init__(self, scorer: str, variant_ids: Sequence[str]):
        self.scorer = scorer
        self.variant_ids = list(variant_ids)
        super().__init__(
            f"{len(self.variant_ids)} variants have no cached {scorer!r} scores "
            f"(first: {self.variant_ids[:5]})"
        )


@dataclass(frozen=True)
class AtlasMatrix:
    """Scores for one scorer: ``values[i, j]`` is variant i on track j."""

    values: np.ndarray
    variant_ids: list[str]
    tracks: pd.DataFrame


def create_client(api_key: Optional[str] = None, timeout: Optional[float] = None):
    """An ``alphagenome`` Atlas client; key from the argument or ``ALPHAGENOME_API_KEY``."""
    api_key = api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"no Atlas API key: pass api_key or set {API_KEY_ENV}")
    from alphagenome.atlas import atlas

    return atlas.create(api_key, timeout=timeout)


def _genome_variant(row):
    from alphagenome.data import genome

    return genome.Variant(
        chromosome=row.chromosome,
        position=int(row.pos),
        reference_bases=row.ref,
        alternate_bases=row.alt,
        name=row.variant_id,
    )


def _id_of(variant) -> str:
    return (
        f"{variant.chromosome}:{int(variant.position)}:"
        f"{variant.reference_bases}>{variant.alternate_bases}"
    )


def _storable(adata):
    """Copy with the ``variant`` objects replaced by a ``variant_id`` string column."""
    out = adata.copy()
    obs = out.obs.copy()
    obs["variant_id"] = [_id_of(v) for v in obs["variant"]]
    obs = obs.drop(columns=["variant"])
    var = out.var.copy()
    # h5ad cannot store object columns holding None; missing text becomes "".
    for frame in (obs, var):
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].where(frame[column].notna(), "").astype(str)
    out.obs = obs
    out.var = var
    return out


def _sha(variant_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(variant_ids).encode()).hexdigest()


def _read_manifest(cache_dir: Path) -> Optional[dict]:
    path = cache_dir / _MANIFEST
    return json.loads(path.read_text()) if path.exists() else None


def _write_manifest(cache_dir: Path, manifest: dict) -> None:
    tmp = cache_dir / (_MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(cache_dir / _MANIFEST)


def _chunk_path(cache_dir: Path, scorer: str, index: int) -> Path:
    return cache_dir / scorer / f"chunk_{index:05d}.h5ad"


def _query_chunk(client, variants: list, scorers: list[str], ontology_terms, max_workers):
    """Scores for a chunk. A ValueError from the batch call is retried per variant."""
    import anndata

    try:
        return dict(client.query_variants(
            variants, requested_scorers=scorers, ontology_terms=ontology_terms,
            progress_bar=False, max_workers=max_workers,
        ))
    except ValueError:
        pass
    parts: dict[str, list] = {name: [] for name in scorers}
    for variant in variants:
        try:
            result = client.query_variant(
                variant, requested_scorers=scorers, ontology_terms=ontology_terms
            )
        except ValueError:
            continue
        for name, adata in result.items():
            parts.setdefault(name, []).append(adata)
    merged = {}
    for name, pieces in parts.items():
        if pieces:
            if any(not piece.var.equals(pieces[0].var) for piece in pieces[1:]):
                raise ValueError(f"inconsistent track metadata for scorer {name!r}")
            combined = anndata.concat(pieces, axis=0, index_unique="-")
            combined.var = pieces[0].var
            combined.obs_names = [str(i) for i in range(combined.n_obs)]
            merged[name] = combined
    return merged


def fetch_atlas_scores(
    client,
    variants: pd.DataFrame,
    requested_scorers: Iterable[str],
    cache_dir,
    *,
    chunk_size: int = 256,
    ontology_terms: Optional[Iterable[str]] = None,
    max_workers: int = 10,
    variant_factory: Optional[Callable[[Any], Any]] = None,
) -> dict:
    """Fetch scores for ``variants`` into ``cache_dir``, resuming finished chunks.

    Args:
        client: an Atlas client (``create_client()``), or any object with the
            same ``query_variants`` / ``query_variant`` methods.
        variants: table with ``chrom``, ``pos`` (1-based), ``ref``, ``alt``.
        requested_scorers: Atlas scorer names; see ``client.scorer_metadata()``.
        cache_dir: directory for the cache; created if absent.
        chunk_size: variants per request batch and per stored file.
        ontology_terms: optional ontology CURIEs to restrict tracks to.
        max_workers: client-side request concurrency.
        variant_factory: builds the client's variant object from a row with
            ``chromosome``, ``pos``, ``ref``, ``alt``, ``variant_id``; defaults
            to ``alphagenome.data.genome.Variant``.

    Returns:
        The manifest. Variants the service could not serve are listed per
        scorer under each chunk's ``missing``; they are never stored as rows.

    Raises:
        ValueError: ``cache_dir`` already holds a fetch with different scorers,
            ontology terms or chunk size.
        Exception: any client error other than ``ValueError`` propagates.
            Chunks finished before it stay on disk and are skipped next time.
    """
    if not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scorers = list(requested_scorers)
    terms = None if ontology_terms is None else list(ontology_terms)
    request = {"requested_scorers": scorers, "ontology_terms": terms,
               "chunk_size": int(chunk_size)}

    manifest = _read_manifest(cache_dir)
    if manifest is None:
        manifest = {**request, "chunks": {}}
    else:
        for key, value in request.items():
            if manifest[key] != value:
                raise ValueError(
                    f"{cache_dir} holds a fetch with {key}={manifest[key]!r}; "
                    f"this call has {key}={value!r}. Use a different cache_dir."
                )

    table = with_variant_ids(variants)
    factory = variant_factory or _genome_variant

    # Invalidate changed/removed chunks before any request or file replacement.
    # Readers only consume committed chunks, including after an interrupted run.
    hashes = {
        str(index): _sha(table.iloc[start:start + chunk_size]["variant_id"].tolist())
        for index, start in enumerate(range(0, len(table), chunk_size))
    }
    stale = [index for index, entry in manifest["chunks"].items()
             if entry["sha"] != hashes.get(index)]
    for index in stale:
        del manifest["chunks"][index]
    _write_manifest(cache_dir, manifest)
    for index in stale:
        for scorer in scorers:
            _chunk_path(cache_dir, scorer, int(index)).unlink(missing_ok=True)

    for index, start in enumerate(range(0, len(table), int(chunk_size))):
        chunk = table.iloc[start:start + int(chunk_size)]
        ids = chunk["variant_id"].tolist()
        entry = manifest["chunks"].get(str(index))
        if (entry and entry["status"] == "done" and entry["sha"] == _sha(ids)
                and all(_chunk_path(cache_dir, scorer, index).exists()
                        or entry["missing"].get(scorer) == ids for scorer in scorers)):
            continue

        manifest["chunks"].pop(str(index), None)
        _write_manifest(cache_dir, manifest)

        results = _query_chunk(
            client, [factory(row) for row in chunk.itertuples(index=False)],
            scorers, terms, max_workers,
        )
        missing = {}
        for scorer in scorers:
            path = _chunk_path(cache_dir, scorer, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            if scorer in results and results[scorer].n_obs:
                stored = _storable(results[scorer])
                tmp = path.with_suffix(".h5ad.tmp")
                try:
                    stored.write_h5ad(tmp)
                    tmp.replace(path)
                finally:
                    tmp.unlink(missing_ok=True)
                served = set(stored.obs["variant_id"])
            else:
                path.unlink(missing_ok=True)
                served = set()
            missing[scorer] = [i for i in ids if i not in served]
        manifest["chunks"][str(index)] = {"sha": _sha(ids), "status": "done",
                                          "missing": missing}
        _write_manifest(cache_dir, manifest)

    return manifest


def missing_variants(cache_dir, scorer: str) -> list[str]:
    """Variants the service could not serve for ``scorer``, in fetch order."""
    manifest = _read_manifest(Path(cache_dir))
    if manifest is None:
        raise FileNotFoundError(f"no Atlas cache at {cache_dir}")
    out: list[str] = []
    for index in sorted(manifest["chunks"], key=int):
        out.extend(manifest["chunks"][index]["missing"].get(scorer, []))
    return out


def load_atlas_matrix(
    cache_dir,
    scorer: str,
    variant_ids: Sequence[str],
    *,
    layer: Optional[str] = None,
    gene_reduce: Optional[str] = None,
) -> AtlasMatrix:
    """Scores for ``scorer`` as a matrix whose rows follow ``variant_ids``.

    Args:
        layer: None for raw scores, or a layer name such as ``"quantiles"``.
        gene_reduce: how to collapse a gene-scoped scorer's several rows per
            variant into one. ``"maxabs"`` keeps, per track, the signed score
            of largest magnitude across genes. None refuses such scorers.

    Raises:
        AtlasScoresMissing: some requested variants have no cached scores.
        KeyError: ``layer`` is not present for this scorer.
        ValueError: several rows per variant and no ``gene_reduce``.
    """
    import anndata

    if gene_reduce not in (None, "maxabs"):
        raise ValueError(f"gene_reduce must be None or 'maxabs', got {gene_reduce!r}")
    cache_dir = Path(cache_dir)
    manifest = _read_manifest(cache_dir)
    files = [] if manifest is None else [
        _chunk_path(cache_dir, scorer, int(index))
        for index, entry in sorted(manifest["chunks"].items(), key=lambda item: int(item[0]))
        if entry["status"] == "done" and _chunk_path(cache_dir, scorer, int(index)).exists()
    ]
    if not files:
        raise AtlasScoresMissing(scorer, list(variant_ids))

    ids: list[str] = []
    blocks: list[np.ndarray] = []
    tracks = None
    for path in files:
        adata = anndata.read_h5ad(path)
        if layer is None:
            block = adata.X
        else:
            if layer not in adata.layers:
                raise KeyError(f"scorer {scorer!r} has no layer {layer!r} in {path.name}")
            block = adata.layers[layer]
        blocks.append(np.asarray(block, dtype=np.float32))
        ids.extend(adata.obs["variant_id"].astype(str).tolist())
        if tracks is None:
            tracks = adata.var.copy()
        elif not tracks.equals(adata.var):
            raise ValueError(f"inconsistent track metadata for scorer {scorer!r} in {path.name}")
    values = np.concatenate(blocks, axis=0)
    row_ids = np.asarray(ids)

    wanted = list(variant_ids)
    available = set(ids)
    absent = list(dict.fromkeys(vid for vid in wanted if vid not in available))
    if absent:
        raise AtlasScoresMissing(scorer, absent)

    rows_of: dict[str, list[int]] = {}
    for i, vid in enumerate(row_ids):
        rows_of.setdefault(vid, []).append(i)
    if gene_reduce is None:
        several = [v for v in wanted if len(rows_of[v]) > 1]
        if several:
            raise ValueError(
                f"scorer {scorer!r} has several rows per variant (e.g. {several[0]}); "
                "pass gene_reduce='maxabs' to collapse them"
            )
        out = values[[rows_of[v][0] for v in wanted]]
    else:
        out = np.empty((len(wanted), values.shape[1]), dtype=np.float32)
        for i, vid in enumerate(wanted):
            block = values[rows_of[vid]]
            pick = np.abs(block).argmax(axis=0)
            out[i] = block[pick, np.arange(block.shape[1])]
    return AtlasMatrix(values=out, variant_ids=wanted, tracks=tracks)
