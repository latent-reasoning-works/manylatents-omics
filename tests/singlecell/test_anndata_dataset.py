"""Tests for AnnDataset (single-cell omics PyTorch dataset)."""
import numpy as np
import pytest
import torch

ad = pytest.importorskip("anndata")
sc = pytest.importorskip("scanpy")

from manylatents.singlecell.data import AnnDataset


@pytest.fixture
def synthetic_h5ad(tmp_path):
    """Create a synthetic AnnData .h5ad file with 100 cells x 50 genes."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 100, 50
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)

    cell_types = np.array(["TypeA", "TypeB", "TypeC"])[rng.integers(0, 3, size=n_cells)]

    import pandas as pd

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": pd.Categorical(cell_types)}, index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )

    path = tmp_path / "synthetic.h5ad"
    adata.write_h5ad(path)
    return path


def test_load_from_path(synthetic_h5ad):
    ds = AnnDataset(adata_path=str(synthetic_h5ad), label_key="cell_type")
    assert ds is not None


def test_shape(synthetic_h5ad):
    ds = AnnDataset(adata_path=str(synthetic_h5ad), label_key="cell_type")
    assert ds.n_samples == 100
    assert ds.n_features == 50


def test_len(synthetic_h5ad):
    ds = AnnDataset(adata_path=str(synthetic_h5ad), label_key="cell_type")
    assert len(ds) == 100


def test_getitem_keys(synthetic_h5ad):
    ds = AnnDataset(adata_path=str(synthetic_h5ad), label_key="cell_type")
    item = ds[0]
    assert isinstance(item, dict)
    assert "data" in item
    assert "metadata" in item


def test_getitem_data_shape(synthetic_h5ad):
    ds = AnnDataset(adata_path=str(synthetic_h5ad), label_key="cell_type")
    item = ds[0]
    assert item["data"].shape == (50,)
    assert item["data"].dtype == torch.float32


def test_getitem_metadata_dtype(synthetic_h5ad):
    ds = AnnDataset(adata_path=str(synthetic_h5ad), label_key="cell_type")
    item = ds[0]
    assert item["metadata"].dtype == torch.long


def test_get_labels(synthetic_h5ad):
    ds = AnnDataset(adata_path=str(synthetic_h5ad), label_key="cell_type")
    labels = ds.get_labels()
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (100,)
    assert labels.dtype == np.int64
    # All codes should be in {0, 1, 2} for 3 categories
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_get_label_names(synthetic_h5ad):
    ds = AnnDataset(adata_path=str(synthetic_h5ad), label_key="cell_type")
    names = ds.get_label_names()
    assert isinstance(names, list)
    assert set(names) == {"TypeA", "TypeB", "TypeC"}


def test_get_data(synthetic_h5ad):
    ds = AnnDataset(adata_path=str(synthetic_h5ad), label_key="cell_type")
    data = ds.get_data()
    assert isinstance(data, torch.Tensor)
    assert data.shape == (100, 50)
    assert data.dtype == torch.float32


def test_no_label_key(synthetic_h5ad):
    """When no label_key is provided, labels should be all zeros."""
    ds = AnnDataset(adata_path=str(synthetic_h5ad))
    labels = ds.get_labels()
    assert np.all(labels == 0)
    assert ds.get_label_names() is None


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        AnnDataset(adata_path="/nonexistent/path.h5ad")


def test_feature_names(synthetic_h5ad):
    ds = AnnDataset(adata_path=str(synthetic_h5ad), label_key="cell_type")
    assert ds.feature_names is not None
    assert len(ds.feature_names) == 50
    assert ds.feature_names[0] == "gene_0"
