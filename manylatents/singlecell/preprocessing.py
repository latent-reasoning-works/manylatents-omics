"""scRNA preprocessing — pure functions over a counts matrix.

Filed as manylatents-omics#60. The consumer is a product layer that owns the ORDER and the
PARAMETERS of a preprocessing workflow and implements none of the compute; what it needs from
here is the operations, each answerable on its own.

Three properties, and each is load-bearing downstream rather than stylistic:

**A filter returns a MASK, never a filtered object.** The consumer keeps the loaded matrix
immutable and carries a selection into it, so that "which cells did this drop, and why" stays
answerable at any later point without having stored a diff. A function that returned a filtered
copy would destroy that and force the caller to reconstruct the difference.

**Nothing mutates its input.** ``sc.pp.normalize_total`` writes IN PLACE, and the matrix a
caller passes is typically the user's own ``.X`` — on a 2700 × 32738 pbmc3k that is an 18.3 MB
CSR. Every scanpy call that could write goes through a copy.

**Sparse in, sparse out where the operation allows.** ``counts > 0`` on a CSR touches stored
entries only; densifying pbmc3k to answer "how many genes per cell" costs 353.6 MB for what is
a column sum. The one operation that cannot avoid it says so at its own docstring.

``detect_doublets`` returns ``is_doublet`` rather than ``keep``, deliberately: the consumer's
default is to FLAG without removing, and inverting a mask is cheaper than recovering a call that
was never made.

**Arrays and names, never AnnData.** The signatures take ``counts`` and ``gene_names`` so the
module is testable on a synthetic matrix and usable by a caller holding something other than an
AnnData. Constructing one internally where scanpy demands it is an implementation detail.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "filter_cells",
    "filter_genes",
    "filter_mito",
    "detect_doublets",
    "normalize",
    "transform",
    "highly_variable_genes",
    "TRANSFORMS",
    "looks_like_counts",
]

#: Transforms this module implements. Closed: a typo'd method silently defaulting to log1p is a
#: caller that believes it applied one thing and applied another.
TRANSFORMS = ("log1p", "sqrt")


def _dense(x: Any) -> np.ndarray:
    """A matrix as a dense ndarray. Only where an operation genuinely cannot stay sparse."""
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _per_cell_genes(counts: Any) -> np.ndarray:
    """Genes detected per cell. ``counts > 0`` on a CSR touches stored entries only, and
    ``np.asarray(...).ravel()`` flattens scipy's ``np.matrix`` result — one path for both types."""
    return np.asarray((counts > 0).sum(axis=1)).ravel()


def _row_sums(counts: Any) -> np.ndarray:
    """Library size per cell, sparse-safe."""
    return np.asarray(counts.sum(axis=1)).ravel()


def _as_adata(counts: Any, gene_names: Optional[Sequence] = None):
    """A COPY of ``counts`` as AnnData, for the scanpy calls that write in place.

    The copy is the whole point — see the module docstring. ``float32`` because scanpy's
    normalization warns on integer dtypes and the caller's counts are usually integral.
    """
    import anndata as ad

    X = counts.copy() if hasattr(counts, "copy") else np.array(counts)
    if hasattr(X, "astype"):
        X = X.astype(np.float32, copy=False)
    adata = ad.AnnData(X)
    if gene_names is not None:
        adata.var_names = [str(g) for g in gene_names]
    return adata


def looks_like_counts(counts: Any, n_rows: int = 100) -> bool:
    """Does this matrix hold integer counts?

    The discriminator a caller needs before normalizing: normalizing an already-normalized matrix
    is silent and wrong, and no downstream number reveals it. Reads the first ``n_rows`` only —
    densifying all of pbmc3k to answer a yes/no question materialises 353.6 MB from an 18.3 MB
    CSR. Answers False for anything it cannot inspect, which is the conservative direction: a
    False leaves the matrix alone.
    """
    try:
        sample = _dense(counts[:n_rows])
        return bool(np.all(sample >= 0) and np.allclose(sample, np.round(sample)))
    except Exception:  # noqa: BLE001 - detection is best-effort by design
        return False


# ── filters: each returns a boolean mask, and removes nothing ────────────────
def filter_cells(
    counts: Any,
    *,
    min_genes: int = 200,
    pct_library_size: Optional[float] = None,
    groups: Optional[Sequence] = None,
) -> np.ndarray:
    """Which cells survive — by genes detected, or by a library-size percentile.

    Two forms because the useful threshold differs by dataset. ``min_genes`` is the Scanpy
    pbmc3k tutorial's number and drops **0** of pbmc3k's 2700 cells, which is why the percentile
    form exists at all. Heumos et al. 2023 (10.1038/s41576-023-00586-w) recommends MAD-based
    outlier detection over fixed cuts; both forms here are the caller's declaration rather than
    a recommendation from this module.

    ``groups`` takes the percentile WITHIN each group — a collection day, typically. Library
    size differs systematically between timepoints, so one global cut drops a whole timepoint
    rather than the worst cells in each.

    Returns a boolean mask over CELLS; ``True`` keeps.
    """
    if pct_library_size is None:
        return _per_cell_genes(counts) >= int(min_genes)

    totals = _row_sums(counts)
    keep = np.ones(totals.shape[0], dtype=bool)
    if groups is None:
        return totals >= np.percentile(totals, float(pct_library_size))
    groups = np.asarray(groups)
    if groups.shape[0] != totals.shape[0]:
        raise ValueError(
            f"groups has {groups.shape[0]} entries for {totals.shape[0]} cells — a per-group "
            "percentile needs one group label per cell")
    for value in np.unique(groups):
        at = groups == value
        keep[at] = totals[at] >= np.percentile(totals[at], float(pct_library_size))
    return keep


def filter_genes(counts: Any, *, min_cells: int = 3) -> np.ndarray:
    """Which genes survive — those detected in at least ``min_cells`` cells.

    Measured on pbmc3k: **19,024** of 32,738 genes are present in fewer than 3 cells. Every
    geometry number computed on the unfiltered matrix is computed over those near-empty columns.

    Returns a boolean mask over GENES; ``True`` keeps.
    """
    per_gene = np.asarray((counts > 0).sum(axis=0)).ravel()
    return per_gene >= int(min_cells)


def filter_mito(
    counts: Any,
    gene_names: Sequence,
    *,
    max_pct: float = 5.0,
    prefix: str = "MT-",
) -> np.ndarray:
    """Which cells survive a mitochondrial-fraction threshold.

    Reads gene NAMES, and **refuses without them** rather than prefix-matching nothing: a run
    reporting "removed 0 cells" from a nameless matrix is indistinguishable from a dataset with
    no dying cells, and that conflation is the reason this takes ``gene_names`` at all.

    Measured on pbmc3k: **13** genes match ``MT-``, median 2.03 % mitochondrial, **57** cells
    above 5 %.

    Returns a boolean mask over CELLS; ``True`` keeps.
    """
    if gene_names is None:
        raise ValueError(
            "filter_mito needs gene names to match a mitochondrial prefix; without them a "
            "prefix match silently selects no genes and reports that it removed no cells")
    names = np.asarray([str(g) for g in gene_names])
    if names.shape[0] != counts.shape[1]:
        raise ValueError(
            f"{names.shape[0]} gene names for {counts.shape[1]} columns — one name per column")

    is_mito = np.array([n.upper().startswith(prefix.upper()) for n in names])
    totals = _row_sums(counts)
    mito = _row_sums(counts[:, is_mito]) if is_mito.any() else np.zeros_like(totals)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(totals > 0, mito / totals * 100.0, 0.0)
    return pct <= float(max_pct)


def detect_doublets(counts: Any, *, threshold: Optional[float] = None,
                    random_state: int = 0) -> np.ndarray:
    """Which cells are likely doublets. Returns ``is_doublet`` — ``True`` means SUSPECT.

    The inverse of the filters above, deliberately: the consumer's default is to flag without
    removing, so the caller inverts when it wants a keep-mask. A call that was never made cannot
    be recovered; an inversion is one operator.

    Densifies, and cannot avoid it — scrublet simulates doublets by summing random cell pairs
    and runs a PCA over the result.
    """
    import scanpy as sc

    adata = _as_adata(counts)
    kwargs: dict[str, Any] = {"random_state": random_state}
    if threshold is not None:
        kwargs["threshold"] = float(threshold)
    sc.pp.scrublet(adata, **kwargs)
    return np.asarray(adata.obs["predicted_doublet"]).astype(bool)


def highly_variable_genes(
    counts: Any,
    *,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
) -> np.ndarray:
    """Which genes are the most variable ACROSS CELLS. Returns a mask over GENES.

    NOT the same statistic as selecting genes by variability across a trajectory's TIME axis —
    that one takes a ``(T, n_genes)`` matrix and answers a different question. They share a name
    and nothing else; if both ever live here they need different ones.

    ``seurat_v3`` expects raw counts and is the flavour that does not require prior
    normalization, which is why it is the default for a function that sits before ``normalize``.
    """
    import scanpy as sc

    adata = _as_adata(counts)
    sc.pp.highly_variable_genes(adata, n_top_genes=int(n_top_genes), flavor=flavor)
    return np.asarray(adata.var["highly_variable"]).astype(bool)


# ── value transforms: a new matrix, nothing removed ──────────────────────────
def normalize(counts: Any, *, target_sum: float = 1e4) -> Any:
    """Library-size normalization — every cell scaled to ``target_sum``. Returns a NEW matrix,
    **of the same kind as the input**: sparse in, sparse out.

    Operates on a copy: ``sc.pp.normalize_total`` writes in place and the caller's matrix is
    typically the user's own ``.X``.

    **Does not check whether the input is already normalized**, because the honest answer to
    that is the caller's: :func:`looks_like_counts` is exported for exactly this, and a
    normalize that silently declined would be as surprising as one that silently ran twice.
    """
    import scanpy as sc

    adata = _as_adata(counts)
    sc.pp.normalize_total(adata, target_sum=float(target_sum))
    # RETURNED AS-IS, sparsity preserved — and NOT through `np.asarray`, which on a scipy sparse
    # matrix WRAPS rather than converts: it returns a 0-dimensional object array holding the
    # matrix, so every downstream shape check silently sees `()`. Measured on a 20 x 6 CSR:
    # `np.asarray(adata.X)` -> shape () dtype object. Densifying instead would be correct and
    # wrong for a different reason — this module's contract is sparse in, sparse out, and
    # pbmc3k densifies from 18.3 MB to 353.6 MB.
    return adata.X


def transform(X: Any, *, method: str = "log1p") -> np.ndarray:
    """``log1p`` or ``sqrt`` over the matrix. Returns a NEW matrix.

    Takes the WORKING matrix rather than the raw counts, so ``normalize`` then ``transform``
    composes: reading the counts here would discard the normalization and transform the raw
    values, in a caller that believes it did both.

    ``sqrt`` is not decoration — it is the PHATE/MIOFlow embryoid-body convention.
    """
    if method not in TRANSFORMS:
        raise ValueError(f"transform: method must be one of {TRANSFORMS}, got {method!r}")
    dense = _dense(X)
    return np.sqrt(dense) if method == "sqrt" else np.log1p(dense)
