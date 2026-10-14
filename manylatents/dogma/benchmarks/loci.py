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


def _position_vector(positions):
    positions = np.asarray(positions)
    if positions.ndim != 1 or positions.dtype.kind not in 'iuf':
        raise ValueError("positions must be a vector of nonnegative integer coordinates")
    if (not np.isfinite(positions).all() or (positions < 0).any()
            or (positions != np.floor(positions)).any()
            or (positions >= 2**63).any()):
        raise ValueError("positions must be nonnegative integer coordinates fitting int64")
    return positions.astype(np.int64)


def _unprefixed_chromosomes(chroms):
    chroms = np.asarray(chroms, dtype=object)
    if chroms.ndim != 1:
        raise ValueError("chromosomes must be a vector")
    return np.asarray([str(c).removeprefix('chr') for c in chroms], dtype=str)


def genomic_blocks(variants, size=100_000) -> np.ndarray:
    """Chromosome/block labels ``chr1:12`` using ``pos // size`` without shifting.

    ``variants`` provides ``chrom`` and ``pos`` columns. Size must be a positive
    integer; positions must be nonnegative integer coordinates fitting int64.
    """
    from numbers import Integral

    if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size <= 0:
        raise ValueError("size must be a positive int")
    chroms = _unprefixed_chromosomes(variants['chrom'])
    positions = _position_vector(variants['pos'])
    if len(chroms) != len(positions):
        raise ValueError("chromosomes and positions must have the same length")
    return np.asarray([f"chr{chrom}:{int(pos) // size}" for chrom, pos in zip(chroms, positions)],
                      dtype=str)


def nearest_site_distance(chroms, positions, site_chroms, site_positions) -> np.ndarray:
    """Nearest same-chromosome site distance in bp, returned as int64.

    Chromosome names are compared after removing a ``chr`` prefix. Sites may be
    unsorted or duplicated. Search is vectorised within each chromosome using
    searchsorted, without a query-by-site distance matrix. A chromosome without
    sites raises MeasurementUnavailable listing all such query chromosomes.
    """
    from manylatents.utils.exceptions import MeasurementUnavailable

    chroms, site_chroms = _unprefixed_chromosomes(chroms), _unprefixed_chromosomes(site_chroms)
    positions, site_positions = _position_vector(positions), _position_vector(site_positions)
    if len(chroms) != len(positions) or len(site_chroms) != len(site_positions):
        raise ValueError("chromosomes and positions must have the same length")
    missing = sorted(set(chroms) - set(site_chroms))
    if missing:
        raise MeasurementUnavailable(f"no sites on query chromosomes: {missing}")
    distances = np.empty(len(positions), dtype=np.int64)
    for chrom in np.unique(chroms):
        rows = chroms == chrom
        queries = positions[rows]
        sites = np.sort(site_positions[site_chroms == chrom])
        insertion = np.searchsorted(sites, queries)
        left = sites[np.maximum(insertion - 1, 0)]
        right = sites[np.minimum(insertion, len(sites) - 1)]
        distances[rows] = np.minimum(np.abs(queries - left), np.abs(queries - right))
    return distances
