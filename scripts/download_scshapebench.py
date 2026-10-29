#!/usr/bin/env python3
"""Download scShapeBench from Hugging Face into ${omics_data:}/scshapebench.

Usage (from a checkout): python -m scripts.download_scshapebench [--subset local|all]
MANYLATENTS_DATA is honored. Re-running resumes; finished files are skipped.

``local`` (default, ~2.6 GB) is labels, synthetic samples and the 9-dataset
sample config, enough to develop on a laptop. ``all`` adds the 102 real
datasets (~116 GB); run it on a cluster node, not a login node.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from manylatents._data_paths import omics_data_root

REPO_ID = "scShape-Benchmark/scShapeBench"
SUBSETS = {
    "local": ["README.md", "labels/*", "data/synthetic/*", "data/sample/*"],
    "all": None,
}


def download(subset: str, output_dir: Path) -> str:
    """Fetch one subset into ``output_dir`` and return the directory."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; expected one of {sorted(SUBSETS)}")
    return snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=SUBSETS[subset],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subset", choices=sorted(SUBSETS), default="local")
    parser.add_argument("--output-dir", type=Path, default=omics_data_root() / "scshapebench")
    args = parser.parse_args()
    print(f"wrote {download(args.subset, args.output_dir)}")


if __name__ == "__main__":
    main()
