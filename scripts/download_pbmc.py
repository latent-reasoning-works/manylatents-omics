#!/usr/bin/env python3
"""
Download and prepare 10X Genomics PBMC datasets.

This script downloads PBMC datasets and converts them to .h5ad format for use
with the manylatents AnnData infrastructure.

Available datasets:
  - 3k:  ~3,000 cells (full dataset)
  - 10k: ~10,000 cells (requires manual download from 10X Genomics)
  - 68k: ~700 cells (reduced/subsampled version, good for quick testing)

Note: The "68k" dataset is a pre-processed reduced version with only 700 cells.
      For the full 68,000-cell dataset, see manual download instructions below.

Usage:
    python scripts/download_pbmc.py --dataset 3k --output-dir data/single_cell
    python scripts/download_pbmc.py --dataset all --output-dir data/single_cell
"""

import argparse
import logging
from pathlib import Path

import scanpy as sc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DATASET_INFO = {
    "3k": {
        "function": sc.datasets.pbmc3k,
        "description": "PBMC 3k dataset (~3,000 cells)",
        "filename": "pbmc_3k.h5ad",
    },
    "68k": {
        "function": sc.datasets.pbmc68k_reduced,
        "description": "PBMC 68k reduced dataset (~700 cells, subsampled from 68k for quick testing)",
        "filename": "pbmc_68k.h5ad",
    },
}


def download_pbmc_10k(output_path):
    """
    Download PBMC 10k dataset from 10X Genomics website.

    Note: scanpy doesn't have a built-in function for PBMC 10k, so we need to
    download it manually from 10X Genomics and convert it.
    """
    logger.warning(
        "PBMC 10k dataset requires manual download from 10X Genomics website:\n"
        "URL: https://www.10xgenomics.com/datasets/10-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0\n"
        "After downloading, convert to .h5ad using:\n"
        "  import scanpy as sc\n"
        "  adata = sc.read_10x_h5('path/to/filtered_feature_bc_matrix.h5')\n"
        "  adata.write_h5ad('{}')".format(output_path)
    )
    logger.info(f"Skipping PBMC 10k download - requires manual download")


def download_dataset(dataset, output_dir, force=False):

    """Download and save a specific PBMC dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "10k":
        output_path = output_dir / "pbmc_10k.h5ad"
        download_pbmc_10k(output_path)
        return

    info = DATASET_INFO[dataset]
    output_path = output_dir / info["filename"]

    if output_path.exists() and not force:
        logger.info(f"Dataset {dataset} already exists at {output_path}. Use --force to re-download.")
        return

    logger.info(f"Downloading {info['description']}...")

    try:
        # Download dataset using scanpy
        logger.info("Downloading from scanpy datasets...")
        adata = info["function"]()

        # Save the raw dataset to .h5ad format
        logger.info(f"Saving dataset to {output_path}...")
        adata.write_h5ad(output_path)

        # Print dataset info
        logger.info(f"Dataset saved successfully!")
        logger.info(f"  Cells: {adata.n_obs}")
        logger.info(f"  Genes: {adata.n_vars}")
        logger.info(f"  Available obs fields: {list(adata.obs.columns)}")
        logger.info(f"  Note: Preprocessing (normalization, clustering, etc.) will be handled by manylatents")

    except Exception as e:
        logger.error(f"Error downloading dataset {dataset}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare 10X Genomics PBMC datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["3k", "10k", "68k", "all"],
        default="3k",
        help="Which PBMC dataset to download (default: 3k)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/single_cell"),
        help="Output directory for downloaded datasets (default: data/single_cell)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset already exists",
    )

    args = parser.parse_args()

    logger.info(f"Output directory: {args.output_dir}")

    if args.dataset == "all":
        datasets_to_download = ["3k", "10k", "68k"]
    else:
        datasets_to_download = [args.dataset]

    for dataset in datasets_to_download:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing PBMC {dataset} dataset")
        logger.info(f"{'='*60}")
        download_dataset(dataset, args.output_dir, args.force)

    logger.info("\n" + "="*60)
    logger.info("Done! Downloaded datasets are ready to use with AnnDataModule.")
    logger.info(f"Location: {args.output_dir}")
    logger.info("="*60)

    # Add note about full PBMC 68k
    logger.info("\n" + "="*60)
    logger.info("NOTE: Full PBMC 68k Dataset")
    logger.info("="*60)
    logger.info("The downloaded 68k dataset is a reduced version with ~700 cells.")
    logger.info("For the full 68,000-cell dataset:")
    logger.info("  1. Download from: https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0")
    logger.info("  2. Extract filtered_gene_bc_matrices_h5.h5")
    logger.info("  3. Convert to h5ad:")
    logger.info("       import scanpy as sc")
    logger.info("       adata = sc.read_10x_h5('filtered_gene_bc_matrices_h5.h5')")
    logger.info("       adata.write_h5ad('data/single_cell/pbmc_68k_full.h5ad')")
    logger.info("="*60)


if __name__ == "__main__":
    main()
