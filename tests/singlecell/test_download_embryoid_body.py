"""Offline acquisition checks against the existing Embryoid Body config."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from scipy import sparse
from scipy.io import savemat

ad = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

import h5py
import manylatents.omics_plugin  # Register the data-path resolver.
from manylatents.singlecell.data import AnnDataset
from scripts import download_embryoid_body as eb


@pytest.mark.parametrize("sparse_input", [False, True])
def test_acquisition_matches_config(tmp_path, monkeypatch, sparse_input):
    monkeypatch.setenv("MANYLATENTS_DATA", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["download_embryoid_body.py"])

    def no_network(*args, **kwargs):
        pytest.fail("Cached acquisition must not access the network")

    monkeypatch.setattr(eb, "urlretrieve", no_network)
    # Include an empty cell and an unexpressed gene: neither may be filtered.
    counts = np.array([[0, 3, 0, 8], [0, 0, 0, 0], [1, 0, 0, 2]])
    genes = ["SOX2", "MT-ND1", "UNEXPRESSED", "NANOG"]
    labels = ["2", "1", "2"]
    output_dir = tmp_path / "single_cell"
    output_dir.mkdir()
    savemat(output_dir / "EBdata.mat", {
        "data": sparse.csc_matrix(counts) if sparse_input else counts,
        "EBgenes_name": np.array(genes, dtype=object)[:, None],
        "cells": np.array(labels, dtype=object)[:, None],
    })
    eb.main()

    config = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "manylatents/singlecell/configs/data/embryoid_body.yaml"
    )
    path = Path(config.adata_path)
    assert path == output_dir / "embryoid_body_raw_counts.h5ad"
    result = ad.read_h5ad(path)
    assert result.shape == counts.shape
    assert result.var_names.tolist() == genes
    assert config.layer == "counts" and config.layer in result.layers
    assert config.use_raw is False
    np.testing.assert_array_equal(result.X.toarray(), counts)
    np.testing.assert_array_equal(result.layers[config.layer].toarray(), counts)
    assert isinstance(result.obs[config.label_key].dtype, pd.CategoricalDtype)
    assert result.obs[config.label_key].tolist() == labels
    assert result.uns["source_url"] == eb.DOWNLOAD_URL
    assert not result.obsm and not result.varm
    assert sorted(p.name for p in output_dir.iterdir()) == [
        "EBdata.mat", "embryoid_body_raw_counts.h5ad",
    ]

    # Distinguish layer selection from the pre-existing silent X fallback.
    with h5py.File(path, "r+") as handle:
        handle["X/data"][:] = -99
    ds = AnnDataset(path, label_key=config.label_key,
                    layer=config.layer, use_raw=config.use_raw)
    np.testing.assert_array_equal(ds.get_data().numpy(), counts)
    assert ds.feature_names == genes
    assert ds.get_labels().dtype == np.int64
    info = ds.get_colormap_info()
    assert info.cmap == "categorical" and info.is_categorical
    assert info.label_names == dict(enumerate(ds.get_label_names()))
    assert [info.label_names[code] for code in ds.get_labels()] == labels
    assert AnnDataset(path, layer=config.layer).get_colormap_info().label_names is None


@pytest.mark.parametrize("bad_value", [-1, 0.5, np.nan, np.inf])
def test_rejects_non_counts(tmp_path, bad_value):
    source = tmp_path / "EBdata.mat"
    savemat(source, {"data": np.array([[0, bad_value], [1, 2]])})
    with pytest.raises(ValueError, match="nonnegative integer counts"):
        eb.write_raw_counts(source, tmp_path)
    assert not (tmp_path / "embryoid_body_raw_counts.h5ad").exists()


def test_rejects_misaligned_metadata(tmp_path):
    source = tmp_path / "EBdata.mat"
    savemat(source, {
        "data": np.zeros((2, 3)),
        "EBgenes_name": np.array(["A", "B"], dtype=object),
        "cells": np.array(["1", "2"], dtype=object),
    })
    with pytest.raises(ValueError, match="dimensions"):
        eb.write_raw_counts(source, tmp_path)
    assert not (tmp_path / "embryoid_body_raw_counts.h5ad").exists()
