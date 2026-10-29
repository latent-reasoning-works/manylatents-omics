"""scShapeBench (arXiv:2605.12662) loaders: plain dicts and arrays, no classes.

Files come from the Hugging Face dataset ``scShape-Benchmark/scShapeBench``
(``scripts/download_scshapebench.py``) under ``${omics_data:}/scshapebench``.
Real datasets are h5ad and load through ``AnnDataModule``
(``configs/data/scshapebench.yaml``); this module covers what that per-cell
module cannot: per-dataset shape labels and the synthetic point clouds.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from manylatents._data_paths import omics_data_root

CLASSES = ("archetypal", "multi_branch", "simple_traj", "clusters")
PUBLISHED_AGGREGATIONS = ("majority", "soft", "confidence_weighted", "union")

# The benchmark's default label set. It ships only in the scShapeBench GitHub
# repo (benchmarks/scrnaseq/labels/high_support_labels.json), so it is rebuilt
# here from the annotations: a class is active when at least 5 annotators chose
# it, and a dataset is kept when any class is active. Checked on 2026-10-29
# against that file: same 79 datasets, same labels, same per-class support.
HIGH_SUPPORT_MIN = 5


def scshapebench_root(root: Optional[Path] = None) -> Path:
    """The dataset directory: ``root`` if given, else ``<omics data>/scshapebench``."""
    return Path(root) if root is not None else omics_data_root() / "scshapebench"


# data/sample holds 9 of the real datasets byte for byte (same LFS hashes as
# data/scRNAseq), so a laptop with only the sample subset finds them there.
REAL_DIRS = ("scRNAseq", "sample")


def real_path(dataset_id: str, root: Optional[Path] = None) -> Path:
    """Path to one real dataset's h5ad, e.g. ``real_path("SCD-0010")``."""
    data = scshapebench_root(root) / "data"
    for folder in REAL_DIRS:
        path = data / folder / f"{dataset_id}.h5ad"
        if path.exists():
            return path
    return data / REAL_DIRS[0] / f"{dataset_id}.h5ad"


def list_real(root: Optional[Path] = None) -> list[str]:
    """Dataset ids of the real h5ad files present locally."""
    data = scshapebench_root(root) / "data"
    return sorted({p.stem for folder in REAL_DIRS for p in (data / folder).glob("*.h5ad")})


def load_shape_labels(aggregation: str = "high_support", root: Optional[Path] = None) -> dict:
    """Per-dataset shape labels as ``{dataset_id: {class: value}}``.

    ``high_support`` gives 0/1 values rebuilt from the annotations; the
    published aggregations give the floats in ``labels/labels.parquet``.
    """
    labels_dir = scshapebench_root(root) / "labels"
    if aggregation == "high_support":
        votes = pd.read_parquet(labels_dir / "annotations.parquet")
        support = votes.groupby("dataset_id")[list(CLASSES)].sum()
        kept = support[support.max(axis=1) >= HIGH_SUPPORT_MIN]
        return {
            dataset_id: {c: int(row[c] >= HIGH_SUPPORT_MIN) for c in CLASSES}
            for dataset_id, row in kept.iterrows()
        }
    if aggregation not in PUBLISHED_AGGREGATIONS:
        raise ValueError(
            f"unknown aggregation {aggregation!r}; "
            f"expected high_support or one of {PUBLISHED_AGGREGATIONS}"
        )
    table = pd.read_parquet(labels_dir / "labels.parquet")
    table = table[table["aggregation"] == aggregation]
    return {
        row["dataset_id"]: {c: float(row[c]) for c in CLASSES}
        for _, row in table.iterrows()
    }


def list_synthetic(root: Optional[Path] = None) -> list[int]:
    """Sample ids of the synthetic samples present locally."""
    folder = scshapebench_root(root) / "data" / "synthetic"
    return sorted(int(p.stem.removeprefix("sample_")) for p in folder.glob("sample_*.npz"))


def load_synthetic(sample_id: int, root: Optional[Path] = None) -> dict:
    """One synthetic sample: noisy ``points``, the true ``graph`` and ``meta``.

    ``graph`` uses the scShapeBench payload keys (``nodes``, ``edges``,
    ``node_positions``) so it can be scored by its evaluator unchanged.
    """
    stem = scshapebench_root(root) / "data" / "synthetic" / f"sample_{sample_id:05d}"
    with np.load(f"{stem}.npz", allow_pickle=False) as arrays:
        nodes = arrays["nodes"].tolist()
        graph = {
            "nodes": nodes,
            "edges": arrays["edges"].tolist(),
            "node_positions": dict(zip(nodes, arrays["positions"].tolist())),
        }
        return {
            "points": arrays["noisy_points"],
            "clean_points": arrays["clean_points"],
            "graph": graph,
            "meta": json.loads(Path(f"{stem}.json").read_text()),
        }
