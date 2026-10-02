"""Canonical variant identifiers.

One string form, ``chr<chrom>:<pos>:<ref>><alt>`` with a 1-based position, so
tables from different sources (TraitGym writes ``1``, AlphaGenome ``chr1``)
join on the same key.
"""
from __future__ import annotations

import pandas as pd

VARIANT_COLUMNS = ("chrom", "pos", "ref", "alt")


def _chromosome(chrom) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def variant_id(chrom, pos, ref, alt) -> str:
    """``chr1:1425822:C>G`` for chrom ``1`` or ``chr1``, position 1425822, C to G."""
    return f"{_chromosome(chrom)}:{int(pos)}:{ref}>{alt}"


def with_variant_ids(variants: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``variants`` with ``chromosome`` and ``variant_id`` columns added.

    Raises:
        ValueError: a required column is absent, or two rows are the same variant.
    """
    missing = [c for c in VARIANT_COLUMNS if c not in variants.columns]
    if missing:
        raise ValueError(f"variants table lacks columns: {missing}")
    out = variants.copy()
    out["chromosome"] = [_chromosome(c) for c in out["chrom"]]
    out["variant_id"] = [
        variant_id(c, p, r, a)
        for c, p, r, a in zip(out["chrom"], out["pos"], out["ref"], out["alt"])
    ]
    duplicated = out["variant_id"][out["variant_id"].duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"duplicate variants: {duplicated[:10]}")
    return out


def lookup_tabix_scores(path: str, variants: pd.DataFrame) -> pd.DataFrame:
    """Per-variant values from a bgzipped, tabix-indexed table of SNV scores.

    The table's first four columns are chromosome, 1-based position, reference
    and alternate allele; every further column is a numeric value. Its last
    ``#`` header line names the value columns; without one they are
    ``value_0``, ``value_1``, ... Chromosomes may be written ``1`` or ``chr1``
    on either side.

    Returns one row per input variant, in input order: ``variant_id``,
    ``found``, ``table_ref``, ``reference_matches`` and the value columns. A
    variant absent from the table has ``found`` False and missing values. A
    variant whose reference allele differs from the table's is still returned,
    with ``reference_matches`` False, so the caller decides what to do with it.

    Raises:
        ImportError: ``pysam`` is not installed.
        ValueError: a required column is absent, or two rows are the same variant.
    """
    try:
        import pysam
    except ImportError as error:
        raise ImportError(
            "lookup_tabix_scores needs pysam: pip install 'manylatents-omics[noncoding]'"
        ) from error

    table = with_variant_ids(variants)
    with pysam.TabixFile(path) as tabix:
        contigs = set(tabix.contigs)
        prefixed = any(c.startswith("chr") for c in contigs)
        names = None
        if tabix.header:
            names = tabix.header[-1].lstrip("#").rstrip("\n").split("\t")[4:]
        rows = []
        for chromosome, pos, ref, alt in zip(
            table["chromosome"], table["pos"], table["ref"], table["alt"]
        ):
            contig = chromosome if prefixed else chromosome[3:]
            pos = int(pos)
            row = {"found": False, "table_ref": None, "reference_matches": pd.NA}
            if contig in contigs:
                for line in tabix.fetch(contig, pos - 1, pos):
                    fields = line.split("\t")
                    if int(fields[1]) != pos or fields[3] != alt:
                        continue
                    values = [float(v) for v in fields[4:]]
                    if names is None:
                        names = [f"value_{i}" for i in range(len(values))]
                    row = {
                        "found": True,
                        "table_ref": fields[2],
                        "reference_matches": fields[2] == ref,
                        **dict(zip(names, values)),
                    }
                    break
            rows.append(row)
    out = pd.DataFrame(rows, columns=["found", "table_ref", "reference_matches", *(names or [])])
    out["reference_matches"] = out["reference_matches"].astype("boolean")
    out.insert(0, "variant_id", table["variant_id"].to_numpy())
    return out
