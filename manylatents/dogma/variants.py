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
