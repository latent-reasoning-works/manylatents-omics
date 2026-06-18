"""
Adapters: convert ecosystem formats to typed kinds at the ingestion edge.

AnnData is accepted here and converted to LabeledArray immediately.
No internal code uses AnnData directly — it stays at the edge.
"""

import logging
from typing import Optional
import numpy as np
import xarray as xr
import scipy.sparse as sp
import sparse
from ...kinds.kinds import LabeledArray
import pandas as pd

logger = logging.getLogger(__name__)


def from_anndata(
    adata,
    coords: dict,
    metadata: Optional[dict] = None,
    use_raw: bool = False,
    layer: Optional[str] = None,
) -> LabeledArray:
    """
    Convert AnnData object to typed LabeledArray kind.

    The expression matrix is selected with the same precedence as AnnDataset:
    ``adata.raw.X`` (if ``use_raw``) → ``adata.layers[layer]`` (if ``layer``
    is given and present) → ``adata.X``.
    """
        
    if use_raw and adata.raw is not None:
        X = adata.raw.X
    elif layer is not None and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X
    
    # xarray cannot wrap a scipy.sparse matrix directly
    # Convert to a pydata ``sparse.COO`` duck array
    if sp.issparse(X):
        data = sparse.COO.from_scipy_sparse(X.tocsr())
    else:
        data = sparse.COO.from_numpy(np.asarray(X))
        
    da = xr.DataArray(
        data,
        dims=["cell", "gene"],
        coords=coords,
        attrs=metadata or {},
    )

    kind = LabeledArray(da)

    logger.info(
        f"Successfully converted to LabeledArray: "
        f"shape={kind._da.shape}, dims={list(kind._da.dims)}"
    )

    return kind

def from_bulk(
    counts: pd.DataFrame,
    metadata: Optional[dict] = None,
) -> LabeledArray:
    """
    Convert a bulk expression matrix to a typed LabeledArray kind.

    ``counts`` is genes × samples (rows indexed by gene id, columns by sample
    id). ``metadata`` is attached as DataArray attributes, the same as
    :func:`from_anndata`.

    The frame must declare its orientation via axis names
    (``index.name == "gene"``, ``columns.name == "sample"``) and carry real
    labels — a default integer ``RangeIndex`` is rejected, since values alone
    cannot distinguish a sample from a gene.
    """
    if counts.index.name != "gene" or counts.columns.name != "sample":
        raise ValueError(
            "from_bulk expects a genes × samples frame with "
            "index.name='gene' and columns.name='sample'; got "
            f"index.name={counts.index.name!r}, columns.name={counts.columns.name!r}"
        )
    if isinstance(counts.index, pd.RangeIndex):
        raise ValueError("gene ids missing: counts.index is a default RangeIndex")
    if isinstance(counts.columns, pd.RangeIndex):
        raise ValueError("sample ids missing: counts.columns is a default RangeIndex")

    gene_ids = counts.index.to_list()
    sample_ids = counts.columns.to_list()

    data = sparse.COO.from_numpy(np.asarray(counts.to_numpy()))

    da = xr.DataArray(
        data,
        dims=["gene", "sample"],
        coords={"gene": gene_ids, "sample": sample_ids},
        attrs=metadata or {},
    )

    kind = LabeledArray(da)
    logger.info(
        f"Successfully converted to LabeledArray: "
        f"shape={kind._da.shape}, dims={list(kind._da.dims)}"
    )

    return kind
    
