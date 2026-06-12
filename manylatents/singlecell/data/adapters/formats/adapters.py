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

logger = logging.getLogger(__name__)


def from_anndata(
    adata,
    coords: dict,
    metadata: Optional[dict] = None,
    use_raw: bool = False,
    layer: Optional[str] = None,
    use_time: bool = False
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

    # TODO: handle time once we encounter time-series data

    da = xr.DataArray(
        data,
        dims=["cell", "gene"],
        coords=coords,
        attrs=metadata or {},
    )

    kind = LabeledArray(da)
    
    kind.validate()

    logger.info(
        f"Successfully converted to LabeledArray: "
        f"shape={kind._da.shape}, dims={list(kind._da.dims)}"
    )

    return kind
