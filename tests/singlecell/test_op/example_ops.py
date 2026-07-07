"""Example ops: how an op consumes a kind and enforces its structural contract.

These are minimal, illustrative ops. The point is that an op declares the named
dims (and coords) it needs *up front* via :meth:`LabeledArray.require` — so a
missing dimension fails cleanly with an actionable message instead of erroring
deep inside a computation. Structure is read from named dims, never guessed by
position.
"""

import logging

from manykinds import LabeledArray

logger = logging.getLogger(__name__)


def temporal_analysis(kind: LabeledArray) -> LabeledArray:
    """Example op that REQUIRES a ``time`` dimension; fails cleanly without it.

    Demonstrates the core principle: ops are typed by the kinds they consume.
    ``require`` validates that the named ``cell``/``gene``/``time`` dims exist,
    raising ``ValueError`` if any is missing.
    """
    kind.require("cell", "gene", "time")  # raises "requires dims [...]" if absent

    logger.info("Running temporal_analysis on %r (%d timepoints)", kind, kind.da.sizes["time"])

    # Placeholder: dummy passthrough on time-resolved data.
    # In real ops: compute trajectories, temporal statistics, etc.
    return LabeledArray(kind.da)


def basic_filter(kind: LabeledArray, min_expression: float = 0.0) -> LabeledArray:
    """Filter low-expression genes. Requires only the ``cell``/``gene`` dims.

    Returns:
        A new ``LabeledArray`` with low-expression genes removed (same dims).
    """
    kind.validate()  # well-formed: wraps a non-empty DataArray
    kind.require("cell", "gene")  # structural contract for this op

    logger.info("Filtering genes with min_expression=%s", min_expression)

    da = kind.da
    gene_means = da.mean(dim="cell")
    high_expr_genes = gene_means >= min_expression
    filtered = da.sel(gene=high_expr_genes)

    logger.info(
        "Filtered %d genes; kept %d genes",
        int((~high_expr_genes).sum()),
        int(high_expr_genes.sum()),
    )

    return LabeledArray(filtered)
