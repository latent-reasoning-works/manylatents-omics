"""Chromosome splits for evaluating variant scores without leakage."""
from __future__ import annotations

import numpy as np

AVI_TRAIN_CHROMOSOMES = frozenset({"1", "4", "7", "8", "10", "13", "15"})
AVI_TUNE_CHROMOSOMES = frozenset({"2", "5", "11", "14", "17", "20", "22", "X"})


def _bare(chrom) -> str:
    chrom = str(chrom)
    return chrom[3:] if chrom.startswith("chr") else chrom


def chromosome_split(chroms, train=AVI_TRAIN_CHROMOSOMES, tune=AVI_TUNE_CHROMOSOMES) -> np.ndarray:
    """``"train"``, ``"tune"`` or ``"report"`` for each chromosome name.

    Names may be written ``1`` or ``chr1``. Anything in neither set is
    ``"report"``. Raises ValueError if a chromosome is in both sets.
    """
    train = {_bare(c) for c in train}
    tune = {_bare(c) for c in tune}
    both = sorted(train & tune)
    if both:
        raise ValueError(f"chromosomes in both train and tune: {both}")
    names = [_bare(c) for c in np.asarray(chroms, dtype=object).tolist()]
    return np.array(
        ["train" if c in train else "tune" if c in tune else "report" for c in names],
        dtype=str,
    )
