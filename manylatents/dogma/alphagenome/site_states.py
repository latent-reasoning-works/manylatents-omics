"""Compute AlphaGenome hidden states at each variant, layer by layer, into resumable shards.

Run with ``python -m manylatents.dogma.alphagenome.site_states --help``. Each
shard folder beneath --out holds, per chunk of variants:

    states/chunk_NNNNN.parquet    variant_id, layer, allele, base, repeat, vector (float16)
    deltas/chunk_NNNNN.parquet    variant_id, layer, repeat, vector (float32, alt - ref)
    features/chunk_NNNNN.parquet  variant_id, layer, repeat, delta_norm, cosine, pr3 (float32)

and skipped.tsv (variants not scored, with the reason), scoring_failed.tsv
(variants that ran out of device memory), manifest.json and timing.json.
Deltas and features come from the float32 states, before the cast to float16.
Checkpoints must already be in the local Hugging Face cache. Only the chrom,
pos, ref and alt columns of --variants are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from manylatents.dogma.alphagenome.model import allele_bases, site_sequences, site_window
from manylatents.dogma.latent_features import cosine, delta_norm, pr3
from manylatents.dogma.score_tracks import shard_slice

KINDS = ("states", "deltas", "features")
FEATURES = ("delta_norm", "cosine", "pr3")
_MANIFEST = "manifest.json"


def read_variants(path) -> pd.DataFrame:
    """chrom, pos, ref, alt from a TSV, with ``chromosome`` and ``variant_id`` added.
    No other column is read."""
    from manylatents.dogma.variants import VARIANT_COLUMNS, with_variant_ids

    table = pd.read_csv(path, sep="\t", usecols=list(VARIANT_COLUMNS),
                        dtype={"chrom": str, "ref": str, "alt": str})
    return with_variant_ids(table)


def prepare_variants(variants: pd.DataFrame, fasta, length: int):
    """Split variants into those that can be scored and those that cannot.

    ``fasta`` is a ``pyfaidx.Fasta`` or a mapping chromosome -> sequence.
    Returns ``(kept, skipped)``; ``skipped`` adds ``reason`` (not_snv,
    missing_chromosome, window_off_chromosome or reference_mismatch) and
    ``fasta_base`` (the genome's base at the site, for a mismatch).
    """
    reasons, fasta_bases = [], []
    for row in variants.itertuples(index=False):
        reason, fasta_base = "", ""
        if len(row.ref) != 1 or len(row.alt) != 1 or row.ref.upper() == row.alt.upper() \
                or not set((row.ref + row.alt).upper()) <= set("ACGT"):
            reason = "not_snv"
        elif row.chromosome not in fasta:
            reason = "missing_chromosome"
        else:
            start, index = site_window(row.pos, 1, length)
            if start < 0 or start + length > len(fasta[row.chromosome]):
                reason = "window_off_chromosome"
            else:
                piece = fasta[row.chromosome][int(row.pos) - 1:int(row.pos)]
                fasta_base = str(getattr(piece, "seq", piece)).upper()
                if fasta_base != row.ref.upper():
                    reason = "reference_mismatch"
                else:
                    fasta_base = ""
        reasons.append(reason)
        fasta_bases.append(fasta_base)
    reasons = np.asarray(reasons, dtype=object)
    mask = reasons == ""
    skipped = variants[~mask].assign(reason=reasons[~mask],
                                     fasta_base=np.asarray(fasta_bases, dtype=object)[~mask])
    return variants[mask], skipped


def _out_of_memory(err: BaseException) -> bool:
    from manylatents.dogma.atlas import LocalScorerClient

    return LocalScorerClient._out_of_memory(err)


def score_variants(model, variants: pd.DataFrame, fasta, length: int, layers, *,
                   alleles: str = "all", repeat: int = 1, window: int = 0,
                   batch_size: int = 1, batch_seconds: Optional[list] = None):
    """Site states for each variant, allele and repeat.

    Every (variant, repeat, allele) is one input sequence; sequences go through
    the model ``batch_size`` at a time, the last batch padded, so one shape is
    compiled. Variants must have passed :func:`prepare_variants`.

    Returns:
        ``(states, failed)``: ``states[variant_id][(allele, repeat)][layer]`` is
        a float32 vector; ``failed`` lists ``(variant_id, reason)`` for
        variants whose batch ran out of device memory (none of their states
        are returned).
    """
    from manylatents.dogma.alphagenome.model import one_hot
    from manylatents.dogma.alphagenome.taps import resolve_layers, site_states

    specs = resolve_layers(layers)
    # Keep one reference window per variant; allele inputs differ from it only
    # at the site, so they are made batch by batch.
    items, windows, site_index = [], {}, None
    for row in variants.itertuples(index=False):
        sites = site_sequences(row.chromosome, row.pos, row.ref, row.alt, fasta, length, alleles)
        if site_index is None:
            site_index = sites.site_index
        assert sites.site_index == site_index  # one length, so one site index
        windows[row.variant_id] = sites.sequences["ref"]
        for r in range(repeat):
            for allele, base in sites.bases.items():
                items.append((row.variant_id, allele, r, base))

    def encode(item):
        onehot = one_hot(windows[item[0]])
        onehot[site_index] = one_hot(item[3])[0]
        return onehot

    states: dict = {}
    failed: dict[str, str] = {}
    for begin in range(0, len(items), batch_size):
        batch = items[begin:begin + batch_size]
        padded = batch + [batch[-1]] * (batch_size - len(batch))
        onehot = np.stack([encode(item) for item in padded])
        tick = time.time()
        memory_error = None
        try:
            out = site_states(model, onehot, site_index, specs, window=window)
        except Exception as err:  # noqa: BLE001 - only device-memory failures are caught
            if not _out_of_memory(err):
                raise
            memory_error = str(err).splitlines()[0][:200]
        if memory_error is not None:
            # The traceback of the failed call held its device buffers; keep
            # only the message, and collect, so the next batch has the memory.
            import gc

            gc.collect()
            for item in batch:
                failed.setdefault(item[0], memory_error)
            continue
        if batch_seconds is not None:
            batch_seconds.append(time.time() - tick)
        for i, (vid, allele, r, _) in enumerate(batch):
            states.setdefault(vid, {})[(allele, r)] = {name: out[name][i] for name in out}
    for vid in failed:
        states.pop(vid, None)
    return states, sorted(failed.items())


def _vector_column(vectors, dtype):
    import pyarrow as pa

    lengths = np.fromiter((len(v) for v in vectors), dtype=np.int32, count=len(vectors))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    flat = (np.concatenate(vectors).astype(dtype) if vectors else np.zeros(0, dtype))
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat))


def chunk_tables(states: dict, bases: dict, layer_names, alleles: str, repeat: int) -> dict:
    """The three tables of one chunk (see the module docstring) as pyarrow tables.

    ``bases[variant_id]`` maps allele name -> base. Features and deltas use the
    float32 states; only the states table is cast to float16.
    """
    import pyarrow as pa

    names = list(allele_bases("A", "C", alleles))
    states_cols = {"variant_id": [], "layer": [], "allele": [], "base": [], "repeat": []}
    state_vectors = []
    delta_cols = {"variant_id": [], "layer": [], "repeat": []}
    delta_vectors = []
    feature_cols = {"variant_id": [], "layer": [], "repeat": [],
                    **{name: [] for name in FEATURES}}
    for vid, by_allele in states.items():
        for layer in layer_names:
            for r in range(repeat):
                h = {allele: by_allele[(allele, r)][layer] for allele in names}
                for allele in names:
                    states_cols["variant_id"].append(vid)
                    states_cols["layer"].append(layer)
                    states_cols["allele"].append(allele)
                    states_cols["base"].append(bases[vid][allele])
                    states_cols["repeat"].append(r)
                    state_vectors.append(h[allele])
                delta_cols["variant_id"].append(vid)
                delta_cols["layer"].append(layer)
                delta_cols["repeat"].append(r)
                delta_vectors.append(h["alt"] - h["ref"])
                feature_cols["variant_id"].append(vid)
                feature_cols["layer"].append(layer)
                feature_cols["repeat"].append(r)
                feature_cols["delta_norm"].append(float(delta_norm(h["ref"], h["alt"])))
                feature_cols["cosine"].append(float(cosine(h["ref"], h["alt"])))
                feature_cols["pr3"].append(float(pr3(h["ref"], h["alt"], h["alt2"], h["alt3"]))
                                           if alleles == "all" else np.nan)

    def table(columns, vectors=None, dtype=None):
        arrays = {key: pa.array(value, type=pa.int16() if key == "repeat" else
                                (pa.float32() if key in FEATURES else pa.string()))
                  for key, value in columns.items()}
        if vectors is not None:
            arrays["vector"] = _vector_column(vectors, dtype)
        return pa.table(arrays)

    return {"states": table(states_cols, state_vectors, np.float16),
            "deltas": table(delta_cols, delta_vectors, np.float32),
            "features": table(feature_cols)}


def _sha(variant_ids) -> str:
    return hashlib.sha256("\n".join(variant_ids).encode()).hexdigest()


def _read_manifest(shard_dir: Path) -> Optional[dict]:
    path = shard_dir / _MANIFEST
    return json.loads(path.read_text()) if path.exists() else None


def _write_manifest(shard_dir: Path, manifest: dict) -> None:
    tmp = shard_dir / (_MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(shard_dir / _MANIFEST)


def _chunk_path(shard_dir: Path, kind: str, index: int) -> Path:
    return shard_dir / kind / f"chunk_{index:05d}.parquet"


def _write_table(table, path: Path) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    try:
        pq.write_table(table, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def run_chunks(model, kept: pd.DataFrame, fasta, shard_dir, request: dict, *,
               batch_size: int = 1, retry_failed: bool = False, log=print) -> dict:
    """Score ``kept`` chunk by chunk into ``shard_dir``, skipping finished chunks.

    ``request`` holds sequence_length, layers, alleles, repeat, window and
    chunk_size, plus anything else that must match on resume (model_version,
    organism). Returns run statistics.

    Raises:
        ValueError: ``shard_dir`` holds a run with a different request.
    """
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(shard_dir)
    if manifest is None:
        manifest = {**request, "chunks": {}}
    for key, value in request.items():
        if manifest.get(key) != value:
            raise ValueError(f"{shard_dir} holds a run with {key}={manifest.get(key)!r}; "
                             f"this call has {key}={value!r}. Use a different --out.")
    chunk_size = request["chunk_size"]
    hashes = {str(index): _sha(kept.iloc[start:start + chunk_size]["variant_id"].tolist())
              for index, start in enumerate(range(0, len(kept), chunk_size))}
    stale = [index for index, entry in manifest["chunks"].items()
             if entry["sha"] != hashes.get(index) or (retry_failed and entry["failed"])]
    for index in stale:
        del manifest["chunks"][index]
    _write_manifest(shard_dir, manifest)
    for index in stale:
        for kind in KINDS:
            _chunk_path(shard_dir, kind, int(index)).unlink(missing_ok=True)

    stats = {"chunks_scored": 0, "chunks_resumed": 0, "n_scored": 0, "failed": [],
             "batch_seconds": [], "score_seconds": 0.0}
    for index, start in enumerate(range(0, len(kept), chunk_size)):
        chunk = kept.iloc[start:start + chunk_size]
        entry = manifest["chunks"].get(str(index))
        if entry and entry["status"] == "done" and entry["sha"] == hashes[str(index)] and (
                entry["n_scored"] == 0
                or all(_chunk_path(shard_dir, kind, index).exists() for kind in KINDS)):
            stats["chunks_resumed"] += 1
            continue
        manifest["chunks"].pop(str(index), None)
        _write_manifest(shard_dir, manifest)

        tick = time.time()
        timed = stats["batch_seconds"] if len(stats["batch_seconds"]) < 6 else None
        states, failed = score_variants(
            model, chunk, fasta, request["sequence_length"], request["layers"],
            alleles=request["alleles"], repeat=request["repeat"], window=request["window"],
            batch_size=batch_size, batch_seconds=timed)
        stats["score_seconds"] += time.time() - tick
        bases = {row.variant_id: allele_bases(row.ref, row.alt, request["alleles"])
                 for row in chunk.itertuples(index=False)}
        if states:
            tables = chunk_tables(states, bases, request["layers"], request["alleles"],
                                  request["repeat"])
            for kind in KINDS:
                _write_table(tables[kind], _chunk_path(shard_dir, kind, index))
        if failed:
            log(f"chunk {index}: {len(failed)} variants out of device memory", flush=True)
        manifest["chunks"][str(index)] = {"sha": hashes[str(index)], "status": "done",
                                          "n_scored": len(states),
                                          "failed": [vid for vid, _ in failed]}
        _write_manifest(shard_dir, manifest)
        stats["chunks_scored"] += 1
        stats["n_scored"] += len(states)
        stats["failed"].extend(failed)
    return stats


def _done_files(shard_dir: Path, kind: str) -> list[Path]:
    manifest = _read_manifest(shard_dir)
    if manifest is None:
        return []
    return [_chunk_path(shard_dir, kind, int(index))
            for index, entry in sorted(manifest["chunks"].items(), key=lambda item: int(item[0]))
            if entry["status"] == "done" and entry["n_scored"]]


def shard_dirs(out) -> list[Path]:
    """The shard folders beneath ``out`` (or ``out`` itself if it is one)."""
    out = Path(out)
    if (out / _MANIFEST).exists():
        return [out]
    return sorted(path.parent for path in out.glob(f"shard_*/{_MANIFEST}"))


def read_site_table(out, kind: str, layer: Optional[str] = None) -> pd.DataFrame:
    """One output table (states, deltas or features) across every finished chunk
    of every shard beneath ``out``, optionally for one layer."""
    import pyarrow.parquet as pq

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    filters = None if layer is None else [("layer", "=", layer)]
    frames = [pq.read_table(path, filters=filters).to_pandas()
              for shard in shard_dirs(out) for path in _done_files(shard, kind)]
    if not frames:
        raise FileNotFoundError(f"no finished {kind} chunks beneath {out}")
    return pd.concat(frames, ignore_index=True)


def delta_matrix(out, layer: str, variant_ids, repeat: int = 0) -> np.ndarray:
    """``(N, D)`` float32 alt - ref deltas at ``layer``, rows in ``variant_ids`` order.

    Raises:
        KeyError: some variants have no delta (skipped, failed or not yet run).
    """
    table = read_site_table(out, "deltas", layer)
    table = table[table["repeat"] == repeat]
    by_id = dict(zip(table["variant_id"], table["vector"]))
    wanted = list(variant_ids)
    absent = [vid for vid in wanted if vid not in by_id]
    if absent:
        raise KeyError(f"{len(absent)} variants have no {layer!r} delta beneath {out} "
                       f"(first: {absent[:5]})")
    return np.stack([np.asarray(by_id[vid], dtype=np.float32) for vid in wanted])


def parse_args(argv=None):
    """Parse options without importing JAX or loading the model."""
    from manylatents.dogma.alphagenome.taps import LAYERS

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variants", required=True, help="TSV with chrom, pos, ref, alt")
    parser.add_argument("--fasta", required=True, help="reference genome FASTA")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--model-version", default="fold_0")
    parser.add_argument("--organism", default="HOMO_SAPIENS")
    parser.add_argument("--hf-cache", help="Hugging Face cache folder (default: the usual one)")
    parser.add_argument("--sequence-length", type=int, default=1_048_576)
    parser.add_argument("--layers", nargs="+", default=list(LAYERS), choices=list(LAYERS),
                        metavar="LAYER", help="default: all 26 trunk layers")
    parser.add_argument("--alleles", choices=["all", "ref_alt"], default="all",
                        help="all = ref, alt, and the other two bases as alt2, alt3")
    parser.add_argument("--repeat", type=int, default=1,
                        help="score each variant this many times (noise floor)")
    parser.add_argument("--window", type=int, default=0,
                        help="average this many bins on each side of the site's bin")
    parser.add_argument("--batch-size", type=int, default=1, help="sequences per forward")
    parser.add_argument("--chunk-size", type=int, default=64, help="variants per output file")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="first N variants of the shard (smoke test)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="score again the chunks that hold variants listed in scoring_failed.tsv")
    args = parser.parse_args(argv)
    for name in ("repeat", "batch_size", "chunk_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.window < 0:
        parser.error("--window must be >= 0")
    return args


def load_local(**kwargs):
    from manylatents.dogma.alphagenome.model import load_local as _load

    return _load(**kwargs)


def main(argv=None) -> int:
    args = parse_args(argv)
    from manylatents.dogma.alphagenome import model as model_module

    if args.sequence_length not in model_module.SUPPORTED_LENGTHS:
        raise SystemExit(f"--sequence-length must be one of {sorted(model_module.SUPPORTED_LENGTHS)}")
    import pyfaidx

    variants = read_variants(args.variants)
    # contiguous shards, so a shard is a stable slice of the file
    variants = variants.iloc[shard_slice(len(variants), args.shard, args.n_shards)]
    if args.limit:
        variants = variants.iloc[:args.limit]
    out = Path(args.out) / f"shard_{args.shard:03d}_of_{args.n_shards:03d}"
    out.mkdir(parents=True, exist_ok=True)

    with pyfaidx.Fasta(args.fasta, sequence_always_upper=True) as fasta:
        kept, skipped = prepare_variants(variants, fasta, args.sequence_length)
        skipped.to_csv(out / "skipped.tsv", sep="\t", index=False)
        print(f"shard {args.shard}/{args.n_shards}: {len(variants)} variants, {len(skipped)} "
              f"skipped before scoring {skipped['reason'].value_counts().to_dict()}", flush=True)
        if kept.empty:
            print("No variants remain after filtering; nothing to score.", flush=True)
            return 0

        import jax

        print("jax devices:", [str(d) for d in jax.devices()], flush=True)
        started = time.time()
        model = load_local(model_version=args.model_version, organism=args.organism,
                           hf_cache=args.hf_cache)
        model.check_length(args.sequence_length)
        load_seconds = time.time() - started
        print(f"model {args.model_version} loaded in {load_seconds:.0f} s", flush=True)

        if args.retry_failed and (out / "scoring_failed.tsv").exists():
            earlier = pd.read_csv(out / "scoring_failed.tsv", sep="\t")
            # The retried variants are listed again only if they fail again.
            earlier.iloc[:0].to_csv(out / "scoring_failed.tsv", sep="\t", index=False)
            print(f"retrying {len(earlier)} variants that ran out of device memory", flush=True)

        request = {"sequence_length": args.sequence_length, "layers": list(args.layers),
                   "alleles": args.alleles, "repeat": args.repeat, "window": args.window,
                   "chunk_size": args.chunk_size, "model_version": args.model_version,
                   "organism": model.organism}
        stats = run_chunks(model, kept, fasta, out, request, batch_size=args.batch_size,
                           retry_failed=args.retry_failed)

    # A resumed run keeps the failures that earlier runs recorded.
    failed = pd.DataFrame(stats["failed"], columns=["variant_id", "reason"])
    if (out / "scoring_failed.tsv").exists():
        failed = pd.concat([pd.read_csv(out / "scoring_failed.tsv", sep="\t"), failed])
    failed.drop_duplicates("variant_id").to_csv(out / "scoring_failed.tsv", sep="\t", index=False)
    _, site_index = site_window(1, 1, args.sequence_length)
    timing = {
        "shard": args.shard, "n_shards": args.n_shards,
        "sequence_length": args.sequence_length, "site_index": site_index,
        "model_version": args.model_version, "organism": model.organism,
        "layers": list(args.layers), "alleles": args.alleles, "repeat": args.repeat,
        "window": args.window, "batch_size": args.batch_size,
        "n_variants": int(len(variants)), "n_skipped": int(len(skipped)),
        "skipped": skipped["reason"].value_counts().to_dict(),
        "n_to_score": int(len(kept)), "n_scored_this_run": stats["n_scored"],
        "n_resource_failures": len(stats["failed"]),
        "chunks_scored": stats["chunks_scored"], "chunks_resumed": stats["chunks_resumed"],
        "load_seconds": round(load_seconds, 1),
        "first_batch_seconds": [round(s, 3) for s in stats["batch_seconds"]],
        "score_seconds": round(stats["score_seconds"], 1),
        "seconds_per_variant": round(stats["score_seconds"] / max(stats["n_scored"], 1), 4),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
