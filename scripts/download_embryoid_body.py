#!/usr/bin/env python3
"""Download + preprocess Embryoid Body from raw 10x files (PHATE tutorial-style).

Source tutorial:
https://github.com/KrishnaswamyLab/PHATE/blob/main/Python/tutorial/EmbryoidBody.ipynb

Pipeline:
1) Download Mendeley archive with raw 10x samples (T0_1A ... T8_9E)
2) Load 10x matrices with Scanpy
3) Per-timepoint library-size filtering (20th to 80th percentile)
4) Filter genes with min_cells=10
5) Normalize to median library size
6) Remove top 10% mitochondrial cells
7) sqrt transform
8) PCA(50) and save as .h5ad for manylatents experiments
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import zipfile
from pathlib import Path

import anndata
import numpy as np
import requests
import scanpy as sc

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

RAW10X_DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/v6n743h5ng/download/1"

SAMPLES = ["T0_1A", "T2_3B", "T4_5C", "T6_7D", "T8_9E"]
TIMEPOINTS = ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"]


def _download_file(url: str, dst: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_and_extract_raw10x(output_dir: Path) -> Path:
    """Download and extract raw 10x folders under output_dir/scRNAseq_raw10x."""
    raw_root = output_dir / "scRNAseq_raw10x"
    raw_root.mkdir(parents=True, exist_ok=True)

    if all((raw_root / sample).exists() for sample in SAMPLES):
        log.info("Raw 10x folders already exist. Skipping download/extraction.")
        return raw_root

    zip_path = raw_root / "v6n743h5ng-1.zip"
    temp_extract = raw_root / "temp_extract"

    log.info(f"Downloading raw 10x archive from {RAW10X_DOWNLOAD_URL}...")
    _download_file(RAW10X_DOWNLOAD_URL, zip_path)
    log.info(f"Downloaded: {zip_path}")

    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    temp_extract.mkdir(parents=True, exist_ok=True)

    log.info("Extracting outer zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_extract)

    nested = temp_extract / "scRNAseq.zip"
    if nested.exists():
        log.info("Extracting nested scRNAseq.zip...")
        with zipfile.ZipFile(nested, "r") as zf:
            zf.extractall(temp_extract)

    sc_folder = temp_extract / "scRNAseq"
    if not sc_folder.exists():
        raise RuntimeError(f"Expected extracted folder not found: {sc_folder}")

    log.info("Moving sample folders to scRNAseq_raw10x/...")
    for item in os.listdir(sc_folder):
        src = sc_folder / item
        dst = raw_root / item
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))

    shutil.rmtree(temp_extract)
    if zip_path.exists():
        zip_path.unlink()

    missing = [s for s in SAMPLES if not (raw_root / s).exists()]
    if missing:
        raise RuntimeError(f"Missing expected sample folders after extraction: {missing}")

    return raw_root


def preprocess(raw_root: Path, output_dir: Path, pca_components: int = 50) -> anndata.AnnData:
    """Apply notebook-style preprocessing and save PCA50 h5ad."""
    log.info("Loading raw 10x samples with scanpy.read_10x_mtx...")
    adatas = []
    for sample, timepoint in zip(SAMPLES, TIMEPOINTS):
        ad = sc.read_10x_mtx(
            str(raw_root / sample),
            var_names="gene_symbols",
            make_unique=True,
            cache=True,
        )
        ad.obs["timepoint"] = timepoint
        ad.obs["sample_labels"] = timepoint
        adatas.append(ad)

    adata = sc.concat(adatas, merge="same")
    adata.obs_names_make_unique()
    log.info(f"Raw merged: {adata.n_obs} cells x {adata.n_vars} genes")

    # QC metrics and mitochondrial fraction
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Per-timepoint library-size filtering (20th to 80th percentile)
    cells_to_keep = []
    for tp in TIMEPOINTS:
        sample_mask = adata.obs["timepoint"] == tp
        sample_counts = adata.obs.loc[sample_mask, "total_counts"]
        q20 = np.percentile(sample_counts, 20)
        q80 = np.percentile(sample_counts, 80)
        keep = (sample_counts >= q20) & (sample_counts <= q80)
        cells_to_keep.extend(sample_counts[keep].index.tolist())
    adata = adata[cells_to_keep, :].copy()
    log.info(f"After per-timepoint library filter: {adata.n_obs} cells")

    # Gene and cell filtering
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=np.median(adata.obs["total_counts"]))
    mito_threshold = np.percentile(adata.obs["pct_counts_mt"], 90)
    adata = adata[adata.obs["pct_counts_mt"] < mito_threshold].copy()
    adata.X = np.sqrt(adata.X)
    log.info(f"After tutorial preprocessing: {adata.n_obs} cells x {adata.n_vars} genes")

    # PCA(50) output for manylatents PHATE speed/stability
    sc.tl.pca(adata, n_comps=pca_components, svd_solver="arpack")
    adata_pca = anndata.AnnData(adata.obsm["X_pca"].astype(np.float32), obs=adata.obs.copy())

    h5ad_path = output_dir / "EBT_10x_tutorial_pca50.h5ad"
    npy_path = output_dir / "EBT_10x_tutorial_pca50.npy"
    labels_path = output_dir / "EBT_10x_tutorial_labels.npy"

    anndata.settings.allow_write_nullable_strings = True
    adata_pca.write(str(h5ad_path))
    np.save(str(npy_path), adata_pca.X.astype(np.float32))
    np.save(str(labels_path), adata_pca.obs["timepoint"].values)

    log.info(f"Saved: {h5ad_path} ({adata_pca.n_obs} x {adata_pca.n_vars})")
    log.info(f"Saved: {npy_path}")
    log.info(f"Saved: {labels_path}")
    return adata_pca


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess Embryoid Body from raw 10x")
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
    raw_root = download_and_extract_raw10x(args.output_dir)
    adata = preprocess(raw_root, args.output_dir, pca_components=args.pca_components)

    log.info("\nDone. Output files:")
    log.info(f"  EBT_10x_tutorial_pca50.h5ad ({adata.n_obs} x {adata.n_vars})")


if __name__ == "__main__":
    main()
