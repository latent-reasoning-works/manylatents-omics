"""Read-only descriptive features of a dataset.

AnnData supplies the uns, obsm, var and layers needed by these descriptions;
preprocessing instead operates on arrays and names. No data is loaded here.
The versioned classifier does not replace preprocessing.looks_like_counts.
"""

import numpy as np
from scipy import sparse

COUNTS_LIKE_VERSION = "counts_like/1"
COUNTS_LIKE_MAX_VALUES = 100_000
COUNTS_LIKE_MIN_VALUES = 1_000
COUNTS_LIKE_SEED = 0
COUNTS_LIKE_TOL = 1e-6
COUNTS_LIKE_PROCESSED_FRACTION = 0.01


def _nonzero_values(matrix):
    if sparse.issparse(matrix):
        # Consolidate duplicate coordinates without changing the caller's storage.
        matrix = matrix.tocsr(copy=True)
        matrix.sum_duplicates()
        values = matrix.data
    else:
        values = np.asarray(matrix)
    return values[values != 0]


def counts_like(matrix) -> str:
    """Classify up to 100,000 nonzero values using counts_like/1.

    Fewer than 1,000 values is undefined. Negative values or more than 1%
    nonintegers are processed; all integers are raw_integer; the rest unknown.
    Sampling is uniform without replacement with seed 0. Nonfinite values
    cannot establish integer counts and yield unknown unless negative.
    """
    values = _nonzero_values(matrix)
    if values.size < COUNTS_LIKE_MIN_VALUES:
        return "undefined"
    if values.size > COUNTS_LIKE_MAX_VALUES:
        indices = np.random.default_rng(COUNTS_LIKE_SEED).choice(
            values.size, COUNTS_LIKE_MAX_VALUES, replace=False
        )
        values = values[indices]
    if np.any(values < 0):
        return "processed"
    if not np.all(np.isfinite(values)):
        return "unknown"
    fraction = np.mean(np.abs(values - np.round(values)) > COUNTS_LIKE_TOL)
    if fraction > COUNTS_LIKE_PROCESSED_FRACTION:
        return "processed"
    return "raw_integer" if fraction == 0 else "unknown"


DEFINITION_VERSION = "dataset_features/1"
ROLES = ("sample", "condition", "batch", "time", "labels")
ABSENT = "absent"
STATUSES = ("ok", "not_supplied", "not_in_input", "not_applicable", "undefined")
FIELDS = ("key", "value", "type", "status", "reason", "source", "definition_version")
KEYS = (
    ("n_cells", "int"), ("n_genes", "int"), ("tested_matrix", "str"),
    ("counts_like", "enum"), ("sparsity", "float"), ("modality", "enum"),
    ("technology", "enum"), ("species", "enum"), ("has_mito_genes", "bool"),
    ("technical_batch_column", "str"), ("n_technical_batches", "int"),
    ("sample_column", "str"), ("n_samples", "int"), ("has_batch_column", "bool"),
    ("has_condition_column", "bool"), ("n_condition_levels", "int"),
    ("samples_per_condition_min", "int"), ("has_replicates", "bool"),
    ("has_time_column", "bool"), ("n_timepoints", "int"),
    ("samples_per_timepoint_min", "int"), ("has_cell_labels", "bool"),
    ("batch_condition_confounded", "bool"), ("batch_time_confounded", "bool"),
    ("median_genes_per_cell", "float"), ("spatial_resolution", "enum"),
    ("has_spliced_unspliced_layers", "bool"),
)
_TYPES = dict(KEYS)
_METADATA_KEYS = ("modality", "technology", "species", "spatial_resolution")
_ROLE_KEYS = tuple(key for key, _ in KEYS[9:24])


def _row(key, value=None, *, status="ok", reason=None, source="data"):
    return dict(key=key, value=value if status == "ok" else None,
                type=_TYPES[key], status=status,
                reason=None if status == "ok" else reason, source=source,
                definition_version=DEFINITION_VERSION)


def _data_rows(data, matrix, layer):
    n_cells, n_columns = matrix.shape
    nonzero = matrix != 0
    size = n_cells * n_columns
    classification = counts_like(matrix)
    rows = [
        _row("n_cells", int(n_cells)),
        _row("tested_matrix", "X" if layer is None else f"layers:{layer}"),
        _row("counts_like", classification) if classification != "undefined" else
        _row("counts_like", status="undefined", reason="Fewer than 1,000 nonzero values"),
        _row("sparsity", float(1 - nonzero.sum() / size)) if size else
        _row("sparsity", status="undefined", reason="Matrix has no entries"),
    ]
    gene_keys = ("n_genes", "has_mito_genes", "median_genes_per_cell",
                 "has_spliced_unspliced_layers")
    if data is None:
        rows.extend(_row(key, status="not_applicable", reason="Input has no gene axis")
                    for key in gene_keys)
        return rows
    rows.extend([
        _row("n_genes", int(data.n_vars)),
        _row("has_mito_genes", any(name.startswith(("MT-", "mt-")) for name in data.var_names)),
        _row("has_spliced_unspliced_layers", all(k in data.layers for k in ("spliced", "unspliced"))),
        _row("median_genes_per_cell", float(np.median(np.asarray(nonzero.sum(axis=1)).ravel())))
        if n_cells else _row("median_genes_per_cell", status="undefined", reason="Input has no cells"),
    ])
    return rows


def dataset_features(data, roles, *, layer=None, overrides=None) -> list[dict]:
    """Return 27 typed descriptions without mutating or loading data.

    AnnData has a gene axis; two-dimensional NumPy and SciPy inputs are point
    clouds. Roles map sample, condition, batch, time and labels to obs columns,
    None for not supplied, or 'absent' for confirmed absence. Layer selects X
    by default. Values are native Python objects, ready for JSON encoding.
    """
    for role, column in roles.items():
        if role not in ROLES:
            raise ValueError(f"Unknown role: {role!r}")
        if column is not None and not isinstance(column, str):
            raise ValueError(f"Role {role!r} must be a column name, 'absent', or None")
    if isinstance(data, np.ndarray) or sparse.issparse(data):
        matrix, annotated = data, None
    else:
        from anndata import AnnData

        if not isinstance(data, AnnData):
            raise TypeError("data must be AnnData, a 2-D ndarray, or a 2-D SciPy sparse object")
        matrix, annotated = data.X, data
    if layer is not None:
        if annotated is None or layer not in annotated.layers:
            raise ValueError(f"Unknown layer: {layer!r}")
        matrix = annotated.layers[layer]
    if not (isinstance(matrix, np.ndarray) or sparse.issparse(matrix)) or matrix.ndim != 2:
        raise TypeError("The selected matrix must be a 2-D ndarray or SciPy sparse object")
    rows = _data_rows(annotated, matrix, layer)
    rows.extend(_row(key, status="not_supplied", reason="No role supplied", source="role")
                for key in _ROLE_KEYS)
    rows.extend(_row(key, status="not_applicable" if annotated is None else "not_supplied",
                     reason="Input has no gene axis" if annotated is None else "No metadata supplied",
                     source="metadata") for key in _METADATA_KEYS)
    by_key = {row["key"]: row for row in rows}
    return [by_key[key] for key, _ in KEYS]
