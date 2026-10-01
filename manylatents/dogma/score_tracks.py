"""Compute per-track AlphaGenome variant scores locally into resumable cache shards.

Run with ``python -m manylatents.dogma.score_tracks --help``. Each shard writes
its Atlas cache, prefilter_dropped.tsv and timing.json beneath the output
folder. Checkpoints must already be available in the local Hugging Face cache.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCORERS = ("DNASE", "ATAC", "CHIP_HISTONE", "CHIP_TF", "CAGE", "PROCAP")


def shard_slice(n_rows: int, shard: int, n_shards: int) -> slice:
    """Contiguous shard of rows, covering every row exactly once across shards."""
    if n_rows < 0 or n_shards <= 0 or not 0 <= shard < n_shards:
        raise ValueError("require n_rows >= 0, n_shards > 0 and 0 <= shard < n_shards")
    return slice(n_rows * shard // n_shards, n_rows * (shard + 1) // n_shards)


def prefilter_variants(variants: pd.DataFrame, fasta_path: str, sequence_length: int):
    """Keep variants whose reference allele matches the genome and whose input
    window lies inside the chromosome. Returns (kept, dropped-with-reason)."""
    import pyfaidx

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    reasons = []
    with pyfaidx.Fasta(fasta_path, sequence_always_upper=True) as fasta:
        for row in variants.itertuples(index=False):
            chromosome = row.chromosome
            if chromosome not in fasta:
                reasons.append("missing_chromosome")
                continue
            length = len(fasta[chromosome])
            position0 = int(row.pos) - 1
            # Match reference_interval.resize: centre on the reference allele.
            centre = position0 + (len(row.ref) + 1) // 2
            start = centre - (sequence_length + 1) // 2
            if start < 0 or start + sequence_length > length:
                reasons.append("window_off_chromosome")
            elif fasta[chromosome][position0:position0 + len(row.ref)].seq != row.ref.upper():
                reasons.append("reference_mismatch")
            else:
                reasons.append("")
    reasons = np.asarray(reasons, dtype=str)
    dropped = variants[reasons != ""].assign(reason=reasons[reasons != ""])
    return variants[reasons == ""], dropped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variants", required=True, help="TSV with chrom, pos, ref, alt")
    parser.add_argument("--out", required=True, help="output cache directory")
    parser.add_argument("--fasta", required=True, help="reference genome FASTA")
    parser.add_argument("--model-version", default="fold_0")
    parser.add_argument("--sequence-length", type=int, default=16384,
                        choices=[16384, 131072, 524288, 1048576])
    parser.add_argument("--scorers", nargs="+", default=list(SCORERS))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="first N variants of the shard (smoke test)")
    args = parser.parse_args(argv)

    from manylatents.dogma.atlas import LocalScorerClient, fetch_atlas_scores
    from manylatents.dogma.variants import with_variant_ids

    variants = with_variant_ids(pd.read_csv(args.variants, sep="\t", dtype={"chrom": str}))
    # contiguous shards, so a shard is a stable slice of the file
    variants = variants.iloc[shard_slice(len(variants), args.shard, args.n_shards)]
    if args.limit:
        variants = variants.iloc[:args.limit]
    out = Path(args.out) / f"shard_{args.shard:03d}_of_{args.n_shards:03d}"
    out.mkdir(parents=True, exist_ok=True)

    kept, dropped = prefilter_variants(variants, args.fasta, args.sequence_length)
    dropped.to_csv(out / "prefilter_dropped.tsv", sep="\t", index=False)
    print(f"shard {args.shard}/{args.n_shards}: {len(variants)} variants, "
          f"{len(dropped)} dropped before scoring {dropped['reason'].value_counts().to_dict()}", flush=True)

    if kept.empty:
        print("No variants remain after filtering; nothing to score.", flush=True)
        return 0

    import jax
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome.models import variant_scorers
    from alphagenome_research.model import dna_model as research_model

    print("jax devices:", [str(d) for d in jax.devices()], flush=True)
    started = time.time()
    # Load from the weights already in the local Hugging Face cache. This needs no
    # token and no network; create_from_huggingface() would start an interactive login.
    import huggingface_hub

    checkpoint = huggingface_hub.snapshot_download(
        repo_id=f"google/alphagenome-{args.model_version.replace('_', '-').lower()}",
        local_files_only=True,
    )
    model = research_model.create(
        checkpoint,
        organism_settings={
            ag_dna_model.Organism.HOMO_SAPIENS: research_model.OrganismSettings(
                fasta_path=args.fasta),
        },
    )
    load_seconds = time.time() - started
    print(f"model {args.model_version} loaded in {load_seconds:.0f} s", flush=True)

    client = LocalScorerClient(
        model,
        {name: variant_scorers.RECOMMENDED_VARIANT_SCORERS[name] for name in args.scorers},
        sequence_length=args.sequence_length,
    )

    # Time the first few variants one by one (compilation, then steady state).
    from alphagenome.data import genome

    seconds = []
    for row in kept.head(6).itertuples(index=False):
        variant = genome.Variant(chromosome=row.chromosome, position=int(row.pos),
                                 reference_bases=row.ref, alternate_bases=row.alt,
                                 name=row.variant_id)
        tick = time.time()
        result = client.query_variant(variant, requested_scorers=args.scorers)
        seconds.append(time.time() - tick)
    shapes = {name: list(scored.shape) for name, scored in result.items()}
    example = {name: {"obs_columns": list(scored.obs.columns), "var_columns": list(scored.var.columns),
                      "layers": list(scored.layers.keys()),
                      "abs_max": float(np.abs(scored.X).max())}
               for name, scored in result.items()}
    print(f"per-variant seconds (first six): {[round(s, 2) for s in seconds]}", flush=True)
    print(f"shapes: {shapes}", flush=True)

    started = time.time()
    manifest = fetch_atlas_scores(client, kept, args.scorers, out, chunk_size=args.chunk_size)
    fetch_seconds = time.time() - started
    missing = {name: sum(len(chunk["missing"].get(name, [])) for chunk in manifest["chunks"].values())
               for name in args.scorers}
    timing = {
        "shard": args.shard, "n_shards": args.n_shards,
        "sequence_length": args.sequence_length, "model_version": args.model_version,
        "n_variants": int(len(variants)), "n_prefilter_dropped": int(len(dropped)),
        "n_scored_requested": int(len(kept)), "missing": missing,
        "load_seconds": round(load_seconds, 1),
        "first_variant_seconds": [round(s, 3) for s in seconds],
        "fetch_seconds": round(fetch_seconds, 1),
        "seconds_per_variant": round(fetch_seconds / max(len(kept), 1), 4),
        "shapes": shapes, "example": example,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
