"""Sampling strategies for single-cell data.

``GeosketchSampling`` extends the manylatents SamplingStrategy protocol;
``clone_subsample`` selects whole clones from lineage-traced data.

Geosketch sampling strategy for manylatents.

Extends the manylatents SamplingStrategy protocol with kernel-herding-based
geometric sketching (Hie et al., 2019).  Geosketch selects a maximally
representative subset that preserves rare populations — useful for large
single-cell datasets where uniform random sampling under-represents small
clusters.

Reference:
    Hie, B., Cho, H., DeMeo, B., Bryson, B. & Berger, B.
    Geometric Sketching Compactly Summarizes the Single-Cell
    Transcriptomic Landscape. Cell Systems 8, 483–493 (2019).
"""

import logging
from collections.abc import Sequence
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from manylatents.utils.sampling import _compute_n_samples, _subsample_dataset_metadata

logger = logging.getLogger(__name__)


class GeosketchSampling:
    """Geometric sketching via kernel herding.

    Selects a subset of points that preserves the geometric structure
    of the data, including rare populations that random sampling misses.

    By default operates on ``dataset.data`` (high-dimensional input space),
    which is the standard geosketch usage.  Falls back to ``embeddings``
    (the low-dimensional DR output) when ``dataset.data`` is unavailable.
    Set ``space="embeddings"`` to always sketch in the DR space.
    """

    def __init__(
        self,
        seed: int = 42,
        fraction: Optional[float] = None,
        n_samples: Optional[int] = None,
        space: str = "data",
        replace: bool = False,
    ):
        """
        Args:
            seed: Random seed for reproducibility.
            fraction: Fraction of samples to keep (0, 1].
            n_samples: Absolute number of samples to keep.
            space: Which space to sketch in.
                ``"data"``  — use ``dataset.data`` (high-dim), fall back to embeddings.
                ``"embeddings"`` — always use the DR embeddings.
            replace: If True, sample with replacement.
        """
        self.seed = seed
        self.fraction = fraction
        self.n_samples = n_samples
        self.space = space
        self.replace = replace

    def sample(
        self,
        embeddings: np.ndarray,
        dataset: object,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """Select a geometrically representative subset.

        Returns:
            Tuple of (subsampled_embeddings, subsampled_dataset, indices).
        """
        from geosketch import gs

        n_samples = n_samples if n_samples is not None else self.n_samples
        fraction = fraction if fraction is not None else self.fraction
        seed = seed if seed is not None else self.seed

        total = embeddings.shape[0]
        n = _compute_n_samples(total, n_samples, fraction)

        # Choose the space for geometric sketching
        if self.space == "data" and hasattr(dataset, "data"):
            sketch_input = np.asarray(dataset.data)
            source = "dataset.data"
        else:
            sketch_input = embeddings
            source = "embeddings"
            if self.space == "data":
                logger.warning(
                    "GeosketchSampling: dataset.data unavailable, "
                    "falling back to embeddings"
                )

        logger.info(
            "GeosketchSampling: %d -> %d samples from %s (seed=%d)",
            total,
            n,
            source,
            seed,
        )

        # geosketch.gs returns a list of indices
        indices = gs(sketch_input, n, replace=self.replace, seed=seed)
        indices = np.sort(np.asarray(indices))

        subsampled_embeddings = embeddings[indices]
        subsampled_ds = _subsample_dataset_metadata(dataset, indices)

        return subsampled_embeddings, subsampled_ds, indices


def clone_subsample(
    obs: pd.DataFrame,
    n: int,
    seed: int = 0,
    *,
    time_col: str = "Time point",
    clone_col: str = "clone",
    early: Sequence = (2,),
    late: Sequence = (4, 6),
    min_late: int = 3,
) -> np.ndarray:
    """Subsample clones, not cells, from lineage-traced data.

    A clone is eligible when it has at least one cell at an ``early`` time point
    and at least ``min_late`` cells at the ``late`` time points. Eligible clones
    are visited in ``numpy.random.default_rng(seed).permutation`` order; a clone
    is kept with every one of its cells unless it would push the running cell
    count past ``n``, in which case it is skipped and the next clone is tried.

    Args:
        obs: Per-cell table (for example ``adata.obs``) with the time and clone columns.
        n: Maximum number of cells to return.
        seed: Seed for the clone visiting order.
        time_col: Column holding each cell's time point.
        clone_col: Column holding each cell's clone identifier.
        early: Time points that count as the early (progenitor) sample.
        late: Time points that count as descendants.
        min_late: Minimum number of descendant cells an eligible clone needs.

    Returns:
        Sorted positional indices into ``obs`` of every cell of the chosen clones.
    """
    day = obs[time_col].to_numpy()
    clone = obs[clone_col].to_numpy()
    later = pd.Series(clone[np.isin(day, list(late))]).value_counts()
    early_clones = set(clone[np.isin(day, list(early))])
    eligible = np.array(sorted(c for c in early_clones if later.get(c, 0) >= min_late))
    rng = np.random.default_rng(seed)
    size = pd.Series(clone).value_counts()
    keep, total = [], 0
    for c in rng.permutation(eligible):
        if total + size[c] > n:
            continue
        keep.append(c)
        total += size[c]
    return np.flatnonzero(np.isin(clone, keep))
