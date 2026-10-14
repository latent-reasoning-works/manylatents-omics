"""Chromosome splits for evaluating variant scores without leakage."""
from __future__ import annotations

from numbers import Real

import numpy as np

AVI_TRAIN_CHROMOSOMES = frozenset({"1", "4", "7", "8", "10", "13", "15"})
AVI_TUNE_CHROMOSOMES = frozenset({"2", "5", "11", "14", "17", "20", "22", "X"})


def _bare(chrom) -> str:
    original = chrom
    if isinstance(chrom, Real) and not isinstance(chrom, (bool, np.bool_)):
        if np.isfinite(chrom) and chrom == int(chrom):
            chrom = str(int(chrom))
    if isinstance(chrom, str):
        chrom = chrom.strip()
        if chrom[:3].lower() == "chr":
            chrom = chrom[3:]
        chrom = chrom.upper()
        if chrom in {str(i) for i in range(1, 23)} | {"X", "Y", "M", "MT"}:
            return chrom
    raise ValueError(f"invalid chromosome: {original!r}")


def normalize_chromosomes(chroms) -> np.ndarray:
    """Canonical human chromosome names; reject missing or unrecognised values."""
    values = np.asarray(chroms, dtype=object)
    if values.ndim != 1:
        raise ValueError("chromosomes must be one-dimensional")
    names, invalid = [], []
    for value in values.tolist():
        try:
            names.append(_bare(value))
        except ValueError:
            invalid.append(value)
    if invalid:
        raise ValueError(f"invalid chromosomes: {invalid!r}")
    return np.asarray(names, dtype=str)


def chromosome_split(chroms, train=AVI_TRAIN_CHROMOSOMES, tune=AVI_TUNE_CHROMOSOMES) -> np.ndarray:
    """``"train"``, ``"tune"`` or ``"report"`` for each chromosome name.

    Names accept integral numbers, whitespace and case-insensitive prefixes.
    Valid human chromosomes outside both sets are ``"report"``. Raises
    ValueError for invalid names or a chromosome in both sets.
    """
    train = set(normalize_chromosomes(list(train)))
    tune = set(normalize_chromosomes(list(tune)))
    both = sorted(train & tune)
    if both:
        raise ValueError(f"chromosomes in both train and tune: {both}")
    names = normalize_chromosomes(chroms)
    return np.array(
        ["train" if c in train else "tune" if c in tune else "report" for c in names],
        dtype=str,
    )
