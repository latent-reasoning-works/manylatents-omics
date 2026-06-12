"""
Example ops demonstrating kind validation.

These are minimal examples showing how ops consume kinds and enforce their
required dims structurally (by reading named dims, not by positional guessing).
"""

import logging

from manylatents.singlecell.data.kinds.kinds import LabeledArray  # Core typed kind

logger = logging.getLogger(__name__)


def require_dims(kind: LabeledArray, *dims: str) -> None:
    """Raise ``ValueError`` unless ``kind`` carries every named dimension.

    Ops call this to declare their structural contract up front, so a missing
    dimension fails cleanly with an actionable message instead of erroring deep
    inside a computation.
    """
    present = set(kind.data.dims)
    missing = set(dims) - present
    if missing:
        raise ValueError(
            f"Op requires dimension(s) {sorted(missing)}. "
            f"Got dims: {list(kind.data.dims)}. "
            f"Available coordinates: {list(kind.data.coords)}"
        )


def temporal_analysis(kind: LabeledArray) -> LabeledArray:
    """Example op that REQUIRES a ``time`` dimension.

    Demonstrates the core principle: ops are typed by the kinds they consume.
    This op validates that a named ``time`` dimension exists, failing cleanly
    if not.

    Raises:
        ValueError: If the ``time`` dimension is missing.
    """
    if "time" not in kind.data.dims:
        raise ValueError(
            f"temporal_analysis requires 'time' dimension. "
            f"Got dims: {list(kind.data.dims)}. "
            f"Available coordinates: {list(kind.data.coords)}"
        )

    logger.info(f"Running temporal_analysis on {kind}")
    logger.info(f"Time axis has {len(kind.data.time)} timepoints")

    # Placeholder: dummy operation on time-resolved data.
    # In real ops: compute trajectories, temporal statistics, etc.
    result = kind.data

    return LabeledArray(result, required_dims={"cell", "gene", "time"})


def basic_filter(kind: LabeledArray, min_expression: float = 0.0) -> LabeledArray:
    """Filter low-expression genes.

    Requires the standard ``cell`` and ``gene`` dimensions and validates the
    input is well-formed before computing.

    Returns:
        Filtered LabeledArray (same dims).
    """
    kind.validate()  # Ensure input is well-formed
    require_dims(kind, "cell", "gene")

    logger.info(f"Filtering genes with min_expression={min_expression}")

    # Simple example: mask genes whose mean expression is below threshold.
    da = kind.data
    gene_means = da.mean(dim="cell")
    high_expr_genes = gene_means >= min_expression

    filtered = da.sel(gene=high_expr_genes)

    logger.info(
        f"Filtered {int((~high_expr_genes).sum())} genes; "
        f"kept {int(high_expr_genes.sum())} genes"
    )

    return LabeledArray(filtered, required_dims={"cell", "gene"})
