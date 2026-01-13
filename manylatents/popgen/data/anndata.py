"""PyTorch Lightning DataModule for AnnData files."""

import logging
from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from .anndata_dataset import AnnDataset

logger = logging.getLogger(__name__)


class AnnDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for AnnData (.h5ad) files.

    Handles single-cell RNA-seq, ATAC-seq, and other omics data
    stored in the AnnData format.
    """

    def __init__(
        self,
        adata_path: str,
        label_key: Optional[str] = None,
        layer: Optional[str] = None,
        use_raw: bool = False,
        batch_size: int = 128,
        test_split: float = 0.2,
        num_workers: int = 0,
        shuffle_traindata: bool = False,
        random_state: int = 42,
        mode: str = "full",
    ):
        """
        Initialize the AnnDataModule.

        Parameters
        ----------
        adata_path : str
            Path to the .h5ad file containing the AnnData object.
        label_key : str or None
            Key in adata.obs used as cell labels.
        layer : str or None
            If specified, use adata.layers[layer] instead of adata.X.
        use_raw : bool
            If True, use adata.raw.X instead of adata.X.
        batch_size : int
            Number of samples per batch.
        test_split : float
            Fraction of the dataset to allocate to the test set.
        num_workers : int
            Number of subprocesses for data loading.
        shuffle_traindata : bool
            Whether to shuffle the training data.
        random_state : int
            Random seed for reproducibility.
        mode : str
            'full' uses entire dataset for train/test, 'split' splits the data.
        """
        super().__init__()

        self.adata_path = adata_path
        self.label_key = label_key
        self.layer = layer
        self.use_raw = use_raw
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers
        self.shuffle_traindata = shuffle_traindata
        self.random_state = random_state
        self.mode = mode

        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Prepare data (download, etc.). Called only on rank 0."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training/testing."""
        if self.mode == "full":
            self.train_dataset = AnnDataset(
                adata_path=self.adata_path,
                label_key=self.label_key,
                layer=self.layer,
                use_raw=self.use_raw,
            )
            self.test_dataset = self.train_dataset

        elif self.mode == "split":
            self.dataset = AnnDataset(
                adata_path=self.adata_path,
                label_key=self.label_key,
                layer=self.layer,
                use_raw=self.use_raw,
            )
            test_size = int(len(self.dataset) * self.test_split)
            train_size = len(self.dataset) - test_size

            self.train_dataset, self.test_dataset = random_split(
                self.dataset,
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
