#!/usr/bin/env python3
"""Download + preprocess Embryoid Body from EBdata.mat.

Pipeline (mat-based, adapted to tutorial-style preprocessing):
1) Download EBdata.mat (if needed)
2) Build AnnData from matrix + genes + sample labels
3) Per-timepoint library-size filtering (20th to 80th percentile)
4) Filter genes with min_cells=10
5) Normalize to median library size
6) Remove top 10% mitochondrial cells
7) sqrt transform
8) PCA(50) and save as .h5ad for manylatents experiments
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import anndata
import numpy as np
import requests
import scanpy as sc
from scipy.io import loadmat

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

DOWNLOAD_URLS = [
    "https://raw.githubusercontent.com/KrishnaswamyLab/PHATE/main/data/EBdata.mat",
    "https://github.com/KrishnaswamyLab/PHATE/raw/main/data/EBdata.mat",
]

EXPECTED_SHAPE = (16825, 17580)
TIMEPOINT_MAP = {
    "1": "Day 00-03",
    "2": "Day 06-09",
    "3": "Day 12-15",
    "4": "Day 18-21",
    "5": "Day 24-27",
}


def download_mat(output_dir: Path) -> Path:
    """Download EBdata.mat if missing."""
    mat_path = output_dir / "EBdata.mat"
    if mat_path.exists():
        log.info(f"EBdata.mat already exists at {mat_path}, skipping download")
        return mat_path

    for url in DOWNLOAD_URLS:
        log.info(f"Downloading from {url[:70]}...")
        try:
            r = requests.get(url, allow_redirects=True, timeout=120, stream=True)
            if r.status_code == 200:
                mat_path.write_bytes(r.content)
                size_mb = len(r.content) / 1e6
                md5 = hashlib.md5(r.content).hexdigest()
                log.info(f"Downloaded {size_mb:.1f} MB | MD5: {md5}")
                return mat_path
            log.warning(f"HTTP {r.status_code}")
        except Exception as e:
            log.warning(f"Failed from {url}: {e}")

    raise RuntimeError("Could not download EBdata.mat from any source")


def _extract_labels(cells: np.ndarray) -> np.ndarray:
    labels = []
    for c in cells.flatten():
        try:
            labels.append(str(c[0]))
        except Exception:
            labels.append(str(c))
    return np.asarray(labels)


def preprocess(mat_path: Path, output_dir: Path, pca_components: int = 50) -> anndata.AnnData:
    """Load EBdata.mat and apply mat-based preprocessing."""
    log.info(f"Loading {mat_path}...")
    mat = loadmat(str(mat_path))

    X = mat["data"].astype(np.float32)
    genes = [g[0] for g in mat["EBgenes_name"].flatten()]
    sample_labels = _extract_labels(mat["cells"])
    timepoint = np.array([TIMEPOINT_MAP.get(x, x) for x in sample_labels], dtype=object)

    if X.shape != EXPECTED_SHAPE:
        raise ValueError(f"Shape mismatch: got {X.shape}, expected {EXPECTED_SHAPE}")

    adata = sc.AnnData(X)
    adata.var_names = genes
    adata.obs["sample_labels"] = sample_labels
    adata.obs["timepoint"] = timepoint
    adata.obs_names_make_unique()

    log.info(f"Raw matrix: {adata.n_obs} cells x {adata.n_vars} genes")

    # QC + mito
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Per-timepoint library-size filter
    cells_to_keep = []
    for tp in sorted(adata.obs["timepoint"].astype(str).unique()):
        sample_mask = adata.obs["timepoint"].astype(str) == tp
        sample_counts = adata.obs.loc[sample_mask, "total_counts"]
        q20 = np.percentile(sample_counts, 20)
        q80 = np.percentile(sample_counts, 80)
        keep = (sample_counts >= q20) & (sample_counts <= q80)
        cells_to_keep.extend(sample_counts[keep].index.tolist())
    adata = adata[cells_to_keep, :].copy()
    log.info(f"After per-timepoint library filter: {adata.n_obs} cells")

    # Gene + mito filtering and normalization
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=np.median(adata.obs["total_counts"]))
    mito_threshold = np.percentile(adata.obs["pct_counts_mt"], 90)
    adata = adata[adata.obs["pct_counts_mt"] < mito_threshold].copy()
    adata.X = np.sqrt(adata.X)
    log.info(f"After preprocessing: {adata.n_obs} cells x {adata.n_vars} genes")

    # PCA(50) output for manylatents PHATE speed/stability
    sc.tl.pca(adata, n_comps=pca_components, svd_solver="arpack")
    adata_pca = anndata.AnnData(adata.obsm["X_pca"].astype(np.float32), obs=adata.obs.copy())

    h5ad_path = output_dir / "EBT_mat_tutorial_pca50.h5ad"
    npy_path = output_dir / "EBT_mat_tutorial_pca50.npy"
    labels_path = output_dir / "EBT_mat_tutorial_labels.npy"

    anndata.settings.allow_write_nullable_strings = True
    adata_pca.write(str(h5ad_path))
    np.save(str(npy_path), adata_pca.X.astype(np.float32))
    np.save(str(labels_path), adata_pca.obs["timepoint"].values)

    log.info(f"Saved: {h5ad_path} ({adata_pca.n_obs} x {adata_pca.n_vars})")
    log.info(f"Saved: {npy_path}")
    log.info(f"Saved: {labels_path}")
    return adata_pca


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess Embryoid Body from EBdata.mat")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/single_cell"),
        help="Output directory (default: data/single_cell)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components to save (default: 50)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mat_path = download_mat(args.output_dir)
    adata = preprocess(mat_path, args.output_dir, pca_components=args.pca_components)

    log.info("\nDone. Output files:")
    log.info(f"  EBT_mat_tutorial_pca50.h5ad ({adata.n_obs} x {adata.n_vars})")


if __name__ == "__main__":
    main()
