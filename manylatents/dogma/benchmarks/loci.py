"""Genomic proximity and locus membership for variant benchmarks."""

import numpy as np
import pandas as pd


def genomic_separation(variants: pd.DataFrame) -> np.ndarray:
    """Pairwise bp distances; infinity across chromosomes and on the diagonal."""
    chrom = variants["chrom"].to_numpy()
    pos = variants["pos"].to_numpy(dtype=np.float64)
    separation = np.abs(pos[:, None] - pos[None, :])
    separation[chrom[:, None] != chrom[None, :]] = np.inf
    np.fill_diagonal(separation, np.inf)
    return separation


def chain_loci(variants: pd.DataFrame, gap: int = 1000) -> np.ndarray:
    """Chain positives within ``gap`` bp; controls inherit their match group's locus.

    Requires chrom, pos, label and match_group. A group without a positive
    raises ValueError, as does a missing column.
    """
    for column in ("chrom", "pos", "label", "match_group"):
        if column not in variants:
            raise ValueError(f"variants table lacks column: {column}")
    positives = variants[variants["label"].astype(bool)].sort_values(["chrom", "pos"])
    new_locus = (positives["chrom"] != positives["chrom"].shift()) | (
        positives["pos"] - positives["pos"].shift() > gap
    )
    locus_of_group = dict(zip(positives["match_group"], np.cumsum(new_locus) - 1))
    loci = variants["match_group"].map(locus_of_group)
    if loci.isna().any():
        missing = variants.loc[loci.isna(), "match_group"].unique().tolist()
        raise ValueError(f"match groups have no positive: {missing}")
    return loci.to_numpy(dtype=int)


def positional_scores(variants: pd.DataFrame) -> dict:
    """Three model-free proximity scores (larger means closer or denser).

    Only ``neg_log_nearest_positive_uses_labels`` uses labels, measuring the
    distance to another positive. The other two use cohort positions only.
    No eligible neighbour gives negative infinity for a log-distance score.
    """
    separation = genomic_separation(variants)
    labels = variants["label"].to_numpy(dtype=bool)
    return {
        "n_cohort_within_1kb": (separation <= 1000).sum(axis=1).astype(float),
        "neg_log_nearest_cohort": -np.log10(separation.min(axis=1, initial=np.inf) + 1),
        "neg_log_nearest_positive_uses_labels": -np.log10(
            separation[:, labels].min(axis=1, initial=np.inf) + 1
        ),
    }


def distant_neighbour_distances(distance, separation, exclusion, k):
    """Return sorted (N, k) feature distances after a genomic exclusion.

    ``distance`` is an (N, N) matrix with infinity on its diagonal and is not
    modified. Zero means no restriction; a finite positive exclusion blocks
    neighbours at most that many bp away; infinity permits other chromosomes
    only. Missing neighbours are represented by infinity.
    """
    if exclusion > 0:
        blocked = np.isfinite(separation) if np.isinf(exclusion) else separation <= exclusion
        distance = np.where(blocked, np.inf, distance)
    nearest = np.partition(distance, k - 1, axis=1)[:, :k]
    return np.sort(nearest, axis=1)
