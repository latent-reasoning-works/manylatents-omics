#!/usr/bin/env python3
"""Download and preprocess the Embryoid Body dataset from Moon et al. 2019.

Source: PHATE paper (Nature Biotechnology 2019)
  - Raw data: https://github.com/KrishnaswamyLab/PHATE/tree/main/data
  - Mendeley: https://data.mendeley.com/datasets/v6n743h5ng/1

Preprocessing pipeline:
  1. Download EBdata.mat (16,825 cells × 17,580 genes)
  2. Normalize total counts per cell (target_sum=1e4)
  3. Log1p transform
  4. Select top 2,000 highly variable genes (Seurat v3 method)
  5. PCA to 50 components

Output:
  - EBT_2k_hvg.h5ad: AnnData with 2k HVGs, normalized + log1p
  - EBT_2k_hvg_pca50.npy: (16825, 50) float32 PCA coordinates
  - EBT_2k_hvg_labels.npy: timepoint labels per cell

Usage:
    python scripts/download_embryoid_body.py [--output-dir data/single_cell]

Integrity:
    After download, verify:
    - Raw shape: (16825, 17580)
    - HVG h5ad shape: (16825, 2000)
    - PCA shape: (16825, 50)
    - MD5 of EBdata.mat: (TODO: pin after first verified download)
"""

import argparse
import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import requests
import scanpy as sc
from scipy.io import loadmat

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# Primary: GitHub raw (fast, no redirect)
# Fallback: Zenodo (stable DOI)
DOWNLOAD_URLS = [
    "https://raw.githubusercontent.com/KrishnaswamyLab/PHATE/main/data/EBdata.mat",
    "https://github.com/KrishnaswamyLab/PHATE/raw/main/data/EBdata.mat",
]

EXPECTED_SHAPE = (16825, 17580)
EXPECTED_N_CELLS = 16825


def download_mat(output_dir: Path) -> Path:
    """Download EBdata.mat from PHATE repository."""
    mat_path = output_dir / "EBdata.mat"

    if mat_path.exists():
        log.info(f"EBdata.mat already exists at {mat_path}, skipping download")
        return mat_path

    for url in DOWNLOAD_URLS:
        log.info(f"Downloading from {url[:60]}...")
        try:
            r = requests.get(url, allow_redirects=True, timeout=120, stream=True)
            if r.status_code == 200:
                mat_path.write_bytes(r.content)
                size_mb = len(r.content) / 1e6
                log.info(f"Downloaded: {size_mb:.1f} MB")

                # Compute MD5 for reproducibility tracking
                md5 = hashlib.md5(r.content).hexdigest()
                log.info(f"MD5: {md5}")
                return mat_path
            else:
                log.warning(f"  HTTP {r.status_code}")
        except Exception as e:
            log.warning(f"  Failed: {e}")

    raise RuntimeError("Could not download EBdata.mat from any source")


def preprocess(mat_path: Path, output_dir: Path):
    """Load .mat, preprocess, save h5ad + npy."""
    import anndata

    log.info(f"Loading {mat_path}...")
    mat = loadmat(str(mat_path))

    data = mat["data"]
    genes = [g[0] for g in mat["EBgenes_name"].flatten()]
    log.info(f"Raw data: {data.shape}, genes: {len(genes)}")

    assert data.shape == EXPECTED_SHAPE, (
        f"Shape mismatch: got {data.shape}, expected {EXPECTED_SHAPE}"
    )

    # Build AnnData
    adata = sc.AnnData(data.astype(np.float32))
    adata.var_names = genes

    # Extract timepoint labels
    if "cells" in mat:
        labels = np.array([
            str(c[0]) if hasattr(c, "__len__") else str(c)
            for c in mat["cells"].flatten()
        ])
    else:
        labels = np.array(["unknown"] * data.shape[0])
    adata.obs["sample_labels"] = labels

    # Preprocessing: normalize + log1p + HVG
    log.info("Preprocessing: normalize_total + log1p + HVG(2000)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    log.info(f"HVG selection: {adata_hvg.shape}")

    # Save h5ad
    h5ad_path = output_dir / "EBT_2k_hvg.h5ad"
    anndata.settings.allow_write_nullable_strings = True
    adata_hvg.write(str(h5ad_path))
    log.info(f"Saved: {h5ad_path}")

    # PCA
    log.info("Computing PCA(50)...")
    sc.tl.pca(adata_hvg, n_comps=50, svd_solver="arpack")

    pca_path = output_dir / "EBT_2k_hvg_pca50.npy"
    labels_path = output_dir / "EBT_2k_hvg_labels.npy"
    np.save(str(pca_path), adata_hvg.obsm["X_pca"].astype(np.float32))
    np.save(str(labels_path), adata_hvg.obs["sample_labels"].values)
    log.info(f"Saved: {pca_path} ({adata_hvg.obsm['X_pca'].shape})")
    log.info(f"Saved: {labels_path} ({len(np.unique(labels))} unique labels)")

    # Save HVG gene list for reproducibility verification
    hvg_path = output_dir / "EBT_2k_hvg_genes.txt"
    hvg_path.write_text("\n".join(adata_hvg.var_names.tolist()))
    log.info(f"Saved: {hvg_path} (for integrity verification)")

    return adata_hvg


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess EB dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/single_cell"),
        help="Output directory (default: data/single_cell)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mat_path = download_mat(args.output_dir)
    adata = preprocess(mat_path, args.output_dir)

    log.info(f"\nDone. Output files in {args.output_dir}/:")
    log.info(f"  EBT_2k_hvg.h5ad     ({adata.shape[0]} cells × {adata.shape[1]} genes)")
    log.info(f"  EBT_2k_hvg_pca50.npy ({adata.shape[0]} × 50)")
    log.info(f"  EBT_2k_hvg_labels.npy")
    log.info(f"  EBT_2k_hvg_genes.txt (HVG list for integrity check)")


if __name__ == "__main__":
    main()
