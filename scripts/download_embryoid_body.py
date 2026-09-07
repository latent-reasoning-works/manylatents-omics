#!/usr/bin/env python3
"""Acquire EBdata.mat and write embryoid_body_raw_counts.h5ad.

Usage (from a checkout): python -m scripts.download_embryoid_body [--output-dir PATH]
The default matches ${omics_data:}/single_cell; MANYLATENTS_DATA is honored.
A cached EBdata.mat in the output directory can be converted offline.

Preserves all cells, genes and sample labels supplied by the MAT file, with
untransformed counts in both X and layers['counts']. No QC or reduction is run.
This is the 16,825-cell MAT source used by the existing config, not a
reproduction of the PHATE tutorial starting from 31,161 cells.
"""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat

from manylatents._data_paths import omics_data_root

DOWNLOAD_URL = "https://raw.githubusercontent.com/KrishnaswamyLab/PHATE/main/data/EBdata.mat"


def download_mat(output_dir: Path) -> Path:
    """Reuse the source file, or download it without caching partial transfers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "EBdata.mat"
    if not path.exists():
        partial = path.with_suffix(".mat.part")
        try:
            urlretrieve(DOWNLOAD_URL, partial)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return path


def write_raw_counts(mat_path: Path, output_dir: Path) -> Path:
    """Convert the source matrix and its axes to one counts-only AnnData file."""
    mat = loadmat(mat_path, simplify_cells=True)
    counts = sparse.csr_matrix(mat["data"])
    values = counts.data
    if not (
        np.isfinite(values).all() and (values >= 0).all()
        and (values == np.floor(values)).all()
    ):
        raise ValueError("EBdata.mat must contain finite, nonnegative integer counts")

    genes = np.asarray(mat["EBgenes_name"]).reshape(-1).astype(str)
    labels = np.asarray(mat["cells"]).reshape(-1).astype(str)
    if counts.shape != (len(labels), len(genes)):
        raise ValueError("Count matrix dimensions do not match sample labels and genes")

    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {"sample_labels": pd.Categorical(
                labels, categories=pd.Index(np.unique(labels), dtype=object),
            )},
            index=pd.Index([str(i) for i in range(len(labels))], dtype=object),
        ),
        var=pd.DataFrame(index=pd.Index(genes, dtype=object)),
    )
    adata.layers["counts"] = counts.copy()
    adata.uns["source_url"] = DOWNLOAD_URL
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "embryoid_body_raw_counts.h5ad"
    adata.write_h5ad(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=omics_data_root() / "single_cell",
    )
    args = parser.parse_args()
    path = write_raw_counts(download_mat(args.output_dir), args.output_dir)
    print(f"Saved raw counts: {path}")


if __name__ == "__main__":
    main()
