"""PyTorch Lightning DataModule for CellxGene Census queries."""

import logging
from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


class CensusDataset(Dataset):
    """In-memory dataset from a CellxGene Census query."""

    def __init__(self, data: torch.Tensor, metadata: np.ndarray, label_names: list | None = None):
        self.data = data
        self.metadata = metadata
        self.label_names = label_names
        self.n_samples, self.n_features = data.shape

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "data": self.data[idx],
            "metadata": torch.tensor(self.metadata[idx], dtype=torch.long),
        }

    def get_labels(self) -> np.ndarray:
        return self.metadata

    def get_label_names(self):
        return self.label_names

    def get_data(self) -> torch.Tensor:
        return self.data


class CellxGeneCensusDataModule(LightningDataModule):
    """
    DataModule that queries CellxGene Census for single-cell data.

    Requires: pip install cellxgene-census
    """

    def __init__(
        self,
        organism: str = "Homo sapiens",
        obs_value_filter: str = "tissue_general == 'blood' and disease == 'normal'",
        obs_column_names: Optional[list] = None,
        label_key: str = "cell_type",
        n_cells_max: int = 5000,
        batch_size: int = 128,
        test_split: float = 0.2,
        num_workers: int = 0,
        shuffle_traindata: bool = False,
        random_state: int = 42,
        mode: str = "full",
    ):
        super().__init__()
        self.organism = organism
        self.obs_value_filter = obs_value_filter
        self.obs_column_names = obs_column_names or ["cell_type"]
        self.label_key = label_key
        self.n_cells_max = n_cells_max
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers
        self.shuffle_traindata = shuffle_traindata
        self.random_state = random_state
        self.mode = mode

        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        try:
            import cellxgene_census
        except ImportError:
            raise ImportError(
                "cellxgene-census is required. Install with: pip install cellxgene-census"
            )

        logger.info(f"Querying CellxGene Census: {self.obs_value_filter}")
        with cellxgene_census.open_soma() as census:
            adata = cellxgene_census.get_anndata(
                census,
                organism=self.organism,
                obs_value_filter=self.obs_value_filter,
                obs_column_names=self.obs_column_names,
            )

        logger.info(f"Census returned {adata.shape[0]} cells x {adata.shape[1]} genes")

        # Subsample if too large
        if adata.shape[0] > self.n_cells_max:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(adata.shape[0], self.n_cells_max, replace=False)
            idx.sort()
            adata = adata[idx].copy()
            logger.info(f"Subsampled to {adata.shape[0]} cells")

        # Extract expression matrix
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        data = torch.tensor(X, dtype=torch.float32)

        # Extract labels
        if self.label_key in adata.obs:
            cats = adata.obs[self.label_key].astype("category")
            label_names = cats.cat.categories.tolist()
            metadata = cats.cat.codes.values.astype(np.int64).copy()
        else:
            label_names = None
            metadata = np.zeros(data.shape[0], dtype=np.int64)

        full_dataset = CensusDataset(data, metadata, label_names)

        if self.mode == "full":
            self.train_dataset = full_dataset
            self.test_dataset = full_dataset
        elif self.mode == "split":
            test_size = int(len(full_dataset) * self.test_split)
            train_size = len(full_dataset) - test_size
            self.train_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.random_state),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_traindata,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
