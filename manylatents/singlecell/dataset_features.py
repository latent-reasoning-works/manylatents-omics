"""Read-only descriptive features of a dataset.

AnnData supplies the uns, obsm, var and layers needed by these descriptions;
preprocessing instead operates on arrays and names. No data is loaded here.
The versioned classifier does not replace preprocessing.looks_like_counts.
"""

import re

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


def _row(key, value=None, *, status="ok", reason=None, source="data"):
    return dict(key=key, value=value if status == "ok" else None,
                type=_TYPES[key], status=status,
                reason=None if status == "ok" else reason, source=source,
                definition_version=DEFINITION_VERSION)


def _data_rows(data, matrix, layer):
    n_cells, n_columns = matrix.shape
    # Sparse comparisons may consolidate duplicate entries in place.
    nonzero = (matrix.copy() if sparse.issparse(matrix) else matrix) != 0
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
    rows.extend(_role_rows(annotated, roles))
    rows.extend(_metadata_rows(annotated, overrides or {}))
    by_key = {row["key"]: row for row in rows}
    return [by_key[key] for key, _ in KEYS]


OVERRIDABLE = ("modality", "technology", "species", "spatial_resolution")
MODALITIES = ("scRNA-seq", "snRNA-seq", "scATAC", "multiome", "CITE-seq", "spatial")
TECHNOLOGIES = ("droplet", "plate", "unknown")
SPECIES = ("human", "mouse", "other")
SPATIAL_RESOLUTIONS = ("array_spot", "image_single_cell", "unknown")
PLATFORM_RESOLUTION = {
    **dict.fromkeys(("visium", "visiumhd", "st", "slideseq", "slideseqv2", "hdst"), "array_spot"),
    **dict.fromkeys(("merfish", "seqfish", "seqfishplus", "iss", "xenium", "cosmx"), "image_single_cell"),
}
SPECIES_CASE_FRACTION = 0.8
_ENUMS = dict(modality=MODALITIES, technology=TECHNOLOGIES, species=SPECIES,
              spatial_resolution=SPATIAL_RESOLUTIONS)
_PEAK_NAME = re.compile(r"^chr[0-9A-Za-z_]+[:-][0-9]+-[0-9]+$")


def _modality(data):
    if "spatial" in data.obsm:
        return "spatial"
    feature_types = set(data.var["feature_types"].dropna()) if "feature_types" in data.var else set()
    if {"Peaks", "Gene Expression"} <= feature_types:
        return "multiome"
    if len(data.var_names) and all(_PEAK_NAME.fullmatch(name) for name in data.var_names):
        return "scATAC"
    if "Antibody Capture" in feature_types:
        return "CITE-seq"
    return "scRNA-seq"


def _species(data):
    names = data.var_names
    upper_mito = any(name.startswith("MT-") for name in names)
    lower_mito = any(name.startswith("mt-") for name in names)
    if upper_mito and not lower_mito:
        return "human"
    if lower_mito and not upper_mito:
        return "mouse"
    alphabetic = [name for name in names if name.isalpha()]
    if alphabetic:
        if sum(name.isupper() for name in alphabetic) / len(alphabetic) >= SPECIES_CASE_FRACTION:
            return "human"
        if sum(name[0].isupper() and name[1:].islower() for name in alphabetic) / len(alphabetic) >= SPECIES_CASE_FRACTION:
            return "mouse"
    return None


def _metadata_rows(data, overrides):
    for key, value in overrides.items():
        if key not in OVERRIDABLE:
            raise ValueError(f"Unknown override: {key!r}")
        if not isinstance(value, str) or value not in _ENUMS[key]:
            raise ValueError(f"Invalid override for {key!r}: {value!r}")
    rows = []
    for key in OVERRIDABLE:
        if key in overrides:
            rows.append(_row(key, str(overrides[key]), source="person"))
            continue
        if data is None:
            rows.append(_row(key, status="not_applicable", reason="Input has no gene axis", source="metadata"))
            continue
        value = data.uns.get(key)
        if key != "spatial_resolution" and isinstance(value, str) and value in _ENUMS[key]:
            rows.append(_row(key, str(value), source="metadata"))
        elif key == "modality":
            rows.append(_row(key, _modality(data)))
        elif key == "species":
            value = _species(data)
            rows.append(_row(key, value) if value is not None else
                        _row(key, status="undefined", reason="Gene names do not establish a species"))
        elif key == "technology":
            rows.append(_row(key, "unknown", source="metadata"))
        else:
            platform = data.uns.get("platform", "")
            platform = re.sub(r"[ _-]", "", platform.lower()) if isinstance(platform, str) else ""
            rows.append(_row(key, PLATFORM_RESOLUTION.get(platform, "unknown"), source="metadata"))
    return rows


_ROLE_REQUIREMENTS = {
    "technical_batch_column": ("batch",), "n_technical_batches": ("batch",),
    "sample_column": ("sample",), "n_samples": ("sample",),
    "has_batch_column": ("batch",), "has_condition_column": ("condition",),
    "n_condition_levels": ("condition",), "samples_per_condition_min": ("sample", "condition"),
    "has_replicates": ("sample", "condition"), "has_time_column": ("time",),
    "n_timepoints": ("time",), "samples_per_timepoint_min": ("sample", "time"),
    "has_cell_labels": ("labels",), "batch_condition_confounded": ("batch", "condition"),
    "batch_time_confounded": ("batch", "time"),
}
_ABSENT_VALUES = {
    "batch": {"technical_batch_column": None, "n_technical_batches": 1,
              "has_batch_column": False, "batch_condition_confounded": False,
              "batch_time_confounded": False},
    "sample": {"sample_column": None, "n_samples": 1, "samples_per_condition_min": 0,
               "samples_per_timepoint_min": 0, "has_replicates": False},
    "condition": {"has_condition_column": False, "n_condition_levels": 0},
    "time": {"has_time_column": False, "n_timepoints": 0},
    "labels": {"has_cell_labels": False},
}
_CLUSTER_COLUMN = re.compile(r"cluster|leiden|louvain", re.I)


def _role_rows(data, roles):
    return [_role_row(data, roles, key, needs) for key, needs in _ROLE_REQUIREMENTS.items()]


def _role_row(data, roles, key, needs):
    import pandas as pd

    columns = {role: roles.get(role) for role in needs}
    missing = [role for role, column in columns.items()
               if column not in (None, ABSENT) and (data is None or column not in data.obs)]
    if missing:
        return _row(key, status="not_in_input", source="role",
                    reason=f"Missing column for role: {', '.join(missing)}")
    unspecified = [role for role, column in columns.items() if column is None]
    if unspecified:
        return _row(key, status="not_supplied", source="role",
                    reason=f"Role not supplied: {', '.join(unspecified)}")
    absent = [role for role, column in columns.items() if column == ABSENT]
    if absent:
        for role in absent:
            if key in _ABSENT_VALUES[role]:
                return _row(key, _ABSENT_VALUES[role][key], source="role")
        return _row(key, status="undefined", source="role",
                    reason=f"Required role is absent: {', '.join(absent)}")
    if key in ("technical_batch_column", "sample_column"):
        return _row(key, str(columns[needs[0]]), source="role")

    # Role names remain distinct even when two roles use the same obs column.
    frame = pd.DataFrame({role: data.obs[column] for role, column in columns.items()})
    if len(needs) == 1:
        column = frame[needs[0]]
        levels = int(column.nunique(dropna=True))
        if key in ("n_technical_batches", "n_samples", "n_condition_levels"):
            if column.isna().any():
                return _row(key, status="undefined", source="role", reason="Role column contains missing values")
            value = levels
        elif key == "n_timepoints":
            value = levels
        elif key == "has_time_column":
            numeric = pd.api.types.is_numeric_dtype(column.dtype)
            ordered = isinstance(column.dtype, pd.CategoricalDtype) and column.cat.ordered
            value = bool(levels >= 2 and (numeric or ordered))
        elif key == "has_cell_labels":
            value = levels >= 2 and _CLUSTER_COLUMN.search(columns["labels"]) is None
        else:
            value = levels >= 2
        return _row(key, value, source="role")

    if frame.isna().any().any():
        return _row(key, status="undefined", source="role", reason="Role columns contain missing values")
    if frame.empty:
        return _row(key, status="undefined", source="role", reason="No observed levels")
    level_role = needs[1]
    if needs[0] == "sample":
        minimum = int(frame.groupby(level_role, observed=True)["sample"].nunique().min())
        value = minimum >= 2 if key == "has_replicates" else minimum
    else:
        if frame[level_role].nunique() < 2:
            return _row(key, status="undefined", source="role", reason="Fewer than two observed levels")
        value = bool((frame.groupby("batch", observed=True)[level_role].nunique() == 1).all())
    return _row(key, value, source="role")
