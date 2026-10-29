"""Offline checks for the scShapeBench loaders and download script."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

pytest.importorskip("pyarrow")

import manylatents.omics_plugin  # Register the data-path resolver.
from manylatents.singlecell.data import scshapebench as sb
from scripts import download_scshapebench as dl

CLASSES = ["archetypal", "multi_branch", "simple_traj", "clusters"]


def _annotations(root: Path) -> None:
    # SCD-0001: 6 of 9 say multi_branch, 5 say clusters -> both active.
    # SCD-0002: no class reaches 5 votes -> dropped from high_support.
    rows = []
    for i in range(9):
        rows.append(("SCD-0001", i, 0, int(i < 6), 0, int(i >= 4)))
        rows.append(("SCD-0002", i, int(i < 4), 0, int(i >= 5), 0))
    frame = pd.DataFrame(rows, columns=["dataset_id", "annotator_id", *CLASSES])
    (root / "labels").mkdir(parents=True)
    frame.to_parquet(root / "labels" / "annotations.parquet")
    aggregated = pd.DataFrame(
        [("SCD-0001", "majority", 0.0, 1.0, 0.0, 1.0)],
        columns=["dataset_id", "aggregation", *CLASSES],
    )
    aggregated.to_parquet(root / "labels" / "labels.parquet")


def _synthetic(root: Path, sample_id: int) -> None:
    out = root / "data" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"sample_{sample_id:05d}"
    np.savez(
        f"{stem}.npz",
        nodes=np.arange(3),
        edges=np.array([[0, 1], [1, 2]]),
        positions=np.zeros((3, 2)),
        clean_points=np.ones((5, 2)),
        noisy_points=np.full((5, 2), 2.0),
    )
    Path(f"{stem}.json").write_text(json.dumps({"sample_id": sample_id, "num_components": 1}))


def test_high_support_matches_the_benchmark_rule(tmp_path):
    _annotations(tmp_path)

    labels = sb.load_shape_labels("high_support", root=tmp_path)

    assert labels == {
        "SCD-0001": {"archetypal": 0, "multi_branch": 1, "simple_traj": 0, "clusters": 1}
    }


def test_published_aggregation_is_read_as_is(tmp_path):
    _annotations(tmp_path)

    labels = sb.load_shape_labels("majority", root=tmp_path)

    assert labels["SCD-0001"]["multi_branch"] == 1.0
    with pytest.raises(ValueError, match="unknown aggregation"):
        sb.load_shape_labels("nonsense", root=tmp_path)


def test_load_synthetic_returns_points_graph_and_metadata(tmp_path):
    _synthetic(tmp_path, 7)

    sample = sb.load_synthetic(7, root=tmp_path)

    assert sample["points"].shape == (5, 2)
    assert (sample["points"] == 2.0).all()  # noisy points, not clean
    assert sample["graph"]["edges"] == [[0, 1], [1, 2]]
    assert sample["meta"]["num_components"] == 1
    assert sb.list_synthetic(root=tmp_path) == [7]


def test_root_follows_manylatents_data(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYLATENTS_DATA", str(tmp_path))

    assert sb.scshapebench_root() == tmp_path / "scshapebench"


def test_config_points_at_a_sample_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYLATENTS_DATA", str(tmp_path))
    config = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "manylatents/singlecell/configs/data/scshapebench.yaml"
    )

    sample = tmp_path / "scshapebench" / "data" / "sample"
    sample.mkdir(parents=True)
    (sample / "SCD-0010.h5ad").touch()

    assert Path(config.adata_path) == sample / "SCD-0010.h5ad"
    assert sb.real_path("SCD-0010") == sample / "SCD-0010.h5ad"
    assert sb.list_real() == ["SCD-0010"]
    assert config.label_key == "leiden"


def test_download_subsets_are_patterns(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(dl, "snapshot_download", lambda **kw: calls.update(kw) or kw["local_dir"])

    dl.download("local", tmp_path)

    assert calls["repo_id"] == "scShape-Benchmark/scShapeBench"
    assert calls["repo_type"] == "dataset"
    assert "data/scRNAseq/*" not in calls["allow_patterns"]
    assert "labels/*" in calls["allow_patterns"]
    with pytest.raises(ValueError, match="unknown subset"):
        dl.download("everything", tmp_path)
