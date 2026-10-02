"""ncVarDB pathogenic non-coding variants and benign controls, pinned to a revision.

Only nuclear single-nucleotide variants are returned; excluded rows are counted
in the table's ``attrs``.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from manylatents.dogma.variants import with_variant_ids

REPOSITORY = "Gardner-BinfLab/ncVarDB"
REVISION = "42828402cd3fea565308faaefa187189faf2ec79"
FILES = {"pathogenic": "data/ncVar_pathogenic.tsv", "benign": "data/ncVar_benign.tsv"}
_COLUMNS = ("ID", "genome", "chr", "pos", "ref", "alt", "mut_position")
_BASES = {"A", "C", "G", "T"}


def _download(name: str, cache_dir: Optional[str] = None) -> Path:
    root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "manylatents" / "ncvardb"
    path = root / REVISION / Path(FILES[name]).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{FILES[name]}", partial
        )
        partial.replace(path)
    return path


def read_variants(pathogenic_path, benign_path) -> pd.DataFrame:
    """Pathogenic (``label`` True) and benign SNVs with canonical variant IDs.

    Columns are ``chrom, pos, ref, alt, label, region, source_id, chromosome,
    variant_id``. Exclusion counts are disjoint: ``attrs["n_not_snv"]`` counts
    invalid or unchanged alleles and ``attrs["n_mitochondrial"]`` counts remaining
    SNVs on M/MT (with or without ``chr``).

    Raises ValueError for missing columns, non-hg38 builds or duplicate variants.
    """
    frames = []
    for path, label in ((pathogenic_path, True), (benign_path, False)):
        raw = pd.read_csv(path, sep="\t", dtype=str)
        missing = [c for c in _COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"{path} lacks columns: {missing}")
        if not raw["genome"].eq("hg38").all():
            raise ValueError(f"{path} is not hg38 throughout")
        frames.append(pd.DataFrame({
            "chrom": raw["chr"].str.removeprefix("chr"),
            "pos": raw["pos"].astype(int), "ref": raw["ref"], "alt": raw["alt"],
            "label": label, "region": raw["mut_position"], "source_id": raw["ID"],
        }))
    table = pd.concat(frames, ignore_index=True)
    snv = (table["ref"].isin(_BASES) & table["alt"].isin(_BASES)
           & table["ref"].ne(table["alt"]))
    nuclear = ~table["chrom"].isin(["M", "MT"])
    out = with_variant_ids(table[snv & nuclear]).reset_index(drop=True)
    out.attrs["n_not_snv"] = int((~snv).sum())
    out.attrs["n_mitochondrial"] = int((snv & ~nuclear).sum())
    return out


def load_variants(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Download (once) and read ncVarDB at the pinned revision."""
    return read_variants(_download("pathogenic", cache_dir), _download("benign", cache_dir))
