#!/usr/bin/env python
"""Download canonical scanpy datasets as h5ad for the shapes atlas."""

import os
from pathlib import Path

import anndata
import scanpy as sc

anndata.settings.allow_write_nullable_strings = True

DATA_DIR = Path(__file__).parent.parent / "data" / "single_cell"

DATASETS = {
    "paul15": {
        "loader": lambda: sc.datasets.paul15(),
        "description": "2,730 cells, myeloid differentiation, 7 terminal fates",
    },
    "krumsiek11": {
        "loader": lambda: sc.datasets.krumsiek11(),
        "description": "640 cells, 4-branch simulated lineage (boolean network)",
        "strip_uns": True,
    },
    "pbmc3k_processed": {
        "loader": lambda: sc.datasets.pbmc3k_processed(),
        "description": "2,638 cells, 8 immune clusters, pre-processed with UMAP",
    },
    "toggleswitch": {
        "loader": lambda: sc.datasets.toggleswitch(),
        "description": "200 cells, 2 genes, binary bifurcation",
    },
}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, info in DATASETS.items():
        path = DATA_DIR / f"{name}.h5ad"
        if path.exists():
            print(f"  SKIP {name} (exists at {path})")
            continue

        print(f"  Downloading {name}: {info['description']}")
        try:
            adata = info["loader"]()
            if info.get("strip_uns"):
                adata.uns = {}
            adata.write(path)
            print(f"    Saved: {adata.shape[0]} cells x {adata.shape[1]} genes -> {path}")
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nAll datasets in {DATA_DIR}")


if __name__ == "__main__":
    main()
