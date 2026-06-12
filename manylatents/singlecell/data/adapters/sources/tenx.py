
import logging
from typing import Optional

import pandas as pd
import scanpy as sc

from ..formats.adapters import from_anndata
from ...kinds.kinds import LabeledArray

logger = logging.getLogger(__name__)


def make_data(
    adata_path,
    metadata: Optional[dict] = {},
    use_raw: bool = False,
    layer: Optional[str] = None,
    use_time: bool = False,
) -> LabeledArray:
    """Load a 10x ``.h5`` matrix and convert it to a typed ``LabeledArray``.

    Args:
        adata_path: Path to a 10x ``filtered_feature_bc_matrix.h5`` file.
        metadata: Optional attrs to attach to the array.
        use_raw: Prefer ``adata.raw.X`` for the expression matrix.
        layer: Use ``adata.layers[layer]`` for the expression matrix if present.
        use_time: Extract a per-cell ``time`` coordinate from the cell barcode
            suffix (the trailing ``-N`` group), when present.
    """
    adata = sc.read_10x_h5(adata_path)

    logger.info(
        f"Converting AnnData to LabeledArray "
        f"(use_raw={use_raw}, layer={layer}, use_time={use_time})"
    )

    if adata.obs_names is None or len(adata.obs_names) == 0:
        raise ValueError(f"AnnData at {adata_path} has no obs_names (cell barcodes).")
    if adata.var_names is None or len(adata.var_names) == 0:
        raise ValueError(f"AnnData at {adata_path} has no var_names (gene names).")

    var = adata.var
    gene_ids = var["gene_ids"].values if "gene_ids" in var else None
    if gene_ids is None:
        raise ValueError(
            f"AnnData at {adata_path} is missing the 'gene_ids' column in var; "
            f"available columns: {var.columns.tolist()}"
        )
    feature_types = var["feature_types"].values if "feature_types" in var else None

    if feature_types is not None:
        feature_counts = pd.Series(feature_types).value_counts()
        if not (feature_counts.index == "Gene Expression").all():
            raise ValueError(
                f"Data is not scRNA-seq: found feature types "
                f"{feature_counts.index.tolist()} with counts "
                f"{feature_counts.values.tolist()}"
            )

    coords = {
        "cell": adata.obs_names,
        "gene": adata.var_names,
        "gene_ids": ("gene", gene_ids),
    }

    if use_time:
        try:
            time = adata.obs_names.str.split("-").str[-1]
            coords["time"] = ("cell", time)
        except Exception as e:  # noqa: BLE001 - barcode format varies across datasets
            logger.warning(f"Could not extract time from cell barcodes: {e}")

    if "genome" not in adata.var or len(adata.var) == 0:
        raise ValueError(
            f"AnnData at {adata_path} is missing a non-empty 'genome' column in var; "
            f"available columns: {adata.var.columns.tolist()}"
        )
    metadata["genome"] = adata.var["genome"].iloc[0]

    return from_anndata(
        adata,
        coords=coords,
        metadata=metadata,
        use_raw=use_raw,
        layer=layer,
        use_time=use_time,
    )
