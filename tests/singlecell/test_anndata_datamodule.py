"""Tests for AnnDataModule (PyTorch Lightning DataModule for AnnData)."""
import numpy as np
import pytest

ad = pytest.importorskip("anndata")
sc = pytest.importorskip("scanpy")

from torch.utils.data import DataLoader

from manylatents.singlecell.data import AnnDataModule


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


class TestFullMode:
    """Tests for mode='full' where train and test use the same dataset."""

    def test_setup_creates_datasets(self, synthetic_h5ad):
        dm = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="full",
        )
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.test_dataset is not None

    def test_full_mode_same_dataset(self, synthetic_h5ad):
        dm = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="full",
        )
        dm.setup()
        assert dm.train_dataset is dm.test_dataset

    def test_train_dataloader_returns_dataloader(self, synthetic_h5ad):
        dm = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="full",
            batch_size=32,
        )
        dm.setup()
        loader = dm.train_dataloader()
        assert isinstance(loader, DataLoader)

    def test_batch_keys(self, synthetic_h5ad):
        dm = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="full",
            batch_size=16,
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert "data" in batch
        assert "metadata" in batch

    def test_batch_shapes(self, synthetic_h5ad):
        dm = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="full",
            batch_size=16,
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert batch["data"].shape == (16, 50)
        assert batch["metadata"].shape == (16,)

    def test_val_dataloader(self, synthetic_h5ad):
        dm = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="full",
        )
        dm.setup()
        loader = dm.val_dataloader()
        assert isinstance(loader, DataLoader)

    def test_test_dataloader(self, synthetic_h5ad):
        dm = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="full",
        )
        dm.setup()
        loader = dm.test_dataloader()
        assert isinstance(loader, DataLoader)


class TestSplitMode:
    """Tests for mode='split' where data is split into train and test."""

    def test_setup_creates_separate_datasets(self, synthetic_h5ad):
        dm = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="split",
            test_split=0.2,
        )
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.test_dataset is not None
        assert dm.train_dataset is not dm.test_dataset

    def test_split_sizes(self, synthetic_h5ad):
        dm = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="split",
            test_split=0.2,
        )
        dm.setup()
        assert len(dm.train_dataset) == 80
        assert len(dm.test_dataset) == 20

    def test_split_reproducibility(self, synthetic_h5ad):
        """Same random_state should produce the same split."""
        dm1 = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="split",
            random_state=42,
        )
        dm1.setup()

        dm2 = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="split",
            random_state=42,
        )
        dm2.setup()

        # Both splits should index the same underlying dataset items
        item1 = dm1.train_dataset[0]
        item2 = dm2.train_dataset[0]
        assert (item1["data"] == item2["data"]).all()

    def test_split_train_dataloader(self, synthetic_h5ad):
        dm = AnnDataModule(
            adata_path=str(synthetic_h5ad),
            label_key="cell_type",
            mode="split",
            batch_size=16,
        )
        dm.setup()
        loader = dm.train_dataloader()
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert batch["data"].shape[0] <= 16
        assert batch["data"].shape[1] == 50
