#!/usr/bin/env python3
"""Stage an existing ClinVar consequence-class build into the layout the loader reads.

`ClinVarDataModule` expects `variants.tsv`, `dna.fasta`, `rna.fasta` and
`protein.fasta` in one directory, and it raises pointing at
`scripts/download_clinvar.py` -- a script referenced by the pipeline doc and by
the error message but never committed.

The data it wants already exists, built per consequence class:

    data/clinvar/variants/{missense,splice_donor,synonymous,...}.tsv
    data/clinvar/sequences/missense_{dna,rna,protein}.fasta

The schemas differ, which is why a symlink is not enough:

    built      variation_id  chrom       pos    ref alt gene         clinical_significance label
    expected   variation_id  chromosome  start  stop    gene_symbol  clinical_significance label
                                                        review_status variant_type

`pos` becomes both `start` and `stop`: these classes are point substitutions.
`review_status` and `variant_type` are not in the build, so they are filled with
explicit sentinels rather than invented values -- the loader reads them into
metadata and nothing downstream keys on them.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

EXPECTED = ["variation_id", "gene_symbol", "clinical_significance", "review_status",
            "chromosome", "start", "stop", "variant_type", "label"]


def adapt_rows(source: Path):
    """Yield loader-schema rows from a consequence-class TSV."""
    with source.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = {"variation_id", "chrom", "pos", "gene", "clinical_significance",
                   "label"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{source} lacks columns {sorted(missing)}")
        for row in reader:
            yield {
                "variation_id": row["variation_id"],
                "gene_symbol": row["gene"],
                "clinical_significance": row["clinical_significance"],
                "review_status": "not_provided",
                "chromosome": row["chrom"],
                "start": row["pos"],
                "stop": row["pos"],
                "variant_type": "single_nucleotide_variant",
                "label": row["label"],
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="the built clinvar directory, holding variants/ and sequences/")
    parser.add_argument("--dest", type=Path, required=True,
                        help="directory the loader will be pointed at")
    parser.add_argument("--consequence", default="missense",
                        help="which class to stage (default: missense)")
    args = parser.parse_args()

    variants = args.source / "variants" / f"{args.consequence}.tsv"
    if not variants.is_file():
        raise SystemExit(f"no such consequence table: {variants}")

    args.dest.mkdir(parents=True, exist_ok=True)
    written = 0
    with (args.dest / "variants.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in adapt_rows(variants):
            writer.writerow(row)
            written += 1
    print(f"variants.tsv: {written} rows from {variants.name}")

    # Symlink the sequences rather than copy: the DNA fasta alone is gigabytes,
    # and its .fai index must stay beside it to remain valid.
    for modality in ("dna", "rna", "protein"):
        source = args.source / "sequences" / f"{args.consequence}_{modality}.fasta"
        if not source.is_file():
            print(f"  {modality}.fasta: absent at source, skipped")
            continue
        for suffix in ("", ".fai"):
            src, dst = Path(str(source) + suffix), args.dest / f"{modality}.fasta{suffix}"
            if src.is_file():
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                os.symlink(src, dst)
        print(f"  {modality}.fasta -> {source.name}")


if __name__ == "__main__":
    main()
