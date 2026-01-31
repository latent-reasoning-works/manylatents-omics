"""
ManifoldGeneticsDataModule - Lightning DataModule for manifold-genetics outputs.

This module provides a PyTorch Lightning DataModule that wraps ManifoldGeneticsDataset
and handles train/val/test splits based on train vs test outputs from manifold-genetics.

This replaces the legacy biobank-specific DataModules (HGDPDataModule, AOUDataModule, etc.)
with a single, dataset-agnostic implementation.

Modes:
    - 'split': Use separate train and test CSVs (train_*_path and test_*_path)
    - 'full': Use only train CSVs for both train and test (same data)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .manifold_genetics_dataset import ManifoldGeneticsDataset

logger = logging.getLogger(__name__)


class ManifoldGeneticsDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for manifold-genetics standardized outputs.

    This DataModule is completely biobank-agnostic and replaces legacy
    biobank-specific DataModules. It handles:
    - Loading train and test outputs from manifold-genetics
    - Creating train/val/test DataLoaders
    - Feature selection (which PCs, which K)
    - Batching and multiprocessing

    Train/test splits are determined by manifold-genetics outputs:
    - train_* files → training data
    - test_* files → test data

    Args:
        train_pca_path: Path to train PCA CSV (training samples)
        test_pca_path: Path to test PCA CSV (test samples, only used in 'split' mode)
        train_admixture_paths: Dict mapping K to train admixture paths
        test_admixture_paths: Dict mapping K to test admixture paths (only used in 'split' mode)
        labels_path: Path to labels.csv (must contain all train+test samples)
        colormap_path: Path to colormap.json
        train_embedding_path: Optional path to train embedding CSV
        test_embedding_path: Optional path to test embedding CSV (only used in 'split' mode)
        label_column: Column name in labels.csv to use (default: "Population")
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        mode: 'split' (separate train/test CSVs) or 'full' (train CSVs only, same data for both)
        shuffle_traindata: Whether to shuffle training data

    Example config:
        ```yaml
        _target_: manylatents.popgen.data.ManifoldGeneticsDataModule
        batch_size: 128
        num_workers: 4
        mode: split

        # Paths to manifold-genetics outputs
        train_pca_path: ./data/hgdp/manifold_genetics/pca/train_pca_50.csv
        test_pca_path: ./data/hgdp/manifold_genetics/pca/test_pca_50.csv

        train_admixture_paths:
          5: ./data/hgdp/manifold_genetics/admixture/train.K5.csv
        test_admixture_paths:
          5: ./data/hgdp/manifold_genetics/admixture/test.K5.csv

        labels_path: ./data/hgdp/manifold_genetics/labels.csv
        colormap_path: ./data/hgdp/manifold_genetics/colormap.json
        label_column: Population
        ```
    """
    
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_pca_path: Optional[str] = None,
        test_pca_path: Optional[str] = None,
        train_admixture_paths: Optional[Dict[int, str]] = None,
        test_admixture_paths: Optional[Dict[int, str]] = None,
        labels_path: Optional[str] = None,
        train_labels_path: Optional[str] = None,
        test_labels_path: Optional[str] = None,
        colormap_path: Optional[str] = None,
        train_colormap_path: Optional[str] = None,
        test_colormap_path: Optional[str] = None,
        train_embedding_path: Optional[str] = None,
        test_embedding_path: Optional[str] = None,
        label_column: str = "Population",
        train_label_column: Optional[str] = None,
        test_label_column: Optional[str] = None,
        geographic_labels_path: Optional[str] = None,
        mode: str = "split",
        shuffle_traindata: bool = True,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_pca_path = train_pca_path
        self.test_pca_path = test_pca_path
        self.train_admixture_paths = train_admixture_paths
        self.test_admixture_paths = test_admixture_paths
        self.labels_path = labels_path
        self.train_labels_path = train_labels_path
        self.test_labels_path = test_labels_path
        self.colormap_path = colormap_path
        self.train_colormap_path = train_colormap_path
        self.test_colormap_path = test_colormap_path

        # Validate label paths
        has_unified_labels = labels_path is not None
        has_split_labels = train_labels_path is not None or test_labels_path is not None
        if has_unified_labels and has_split_labels:
            raise ValueError(
                "Cannot specify both labels_path and train_labels_path/test_labels_path. "
                "Use labels_path for 'full' mode or when train+test share labels, "
                "or use train_labels_path/test_labels_path for 'split' mode with different datasets."
            )

        # Validate colormap paths
        has_unified_colormap = colormap_path is not None
        has_split_colormap = train_colormap_path is not None or test_colormap_path is not None
        if has_unified_colormap and has_split_colormap:
            raise ValueError(
                "Cannot specify both colormap_path and train_colormap_path/test_colormap_path. "
                "Use colormap_path for 'full' mode, or use train_colormap_path/test_colormap_path for 'split' mode."
            )
        self.train_embedding_path = train_embedding_path
        self.test_embedding_path = test_embedding_path
        self.label_column = label_column
        self.train_label_column = train_label_column
        self.test_label_column = test_label_column
        self.geographic_labels_path = geographic_labels_path
        self.mode = mode
        self.shuffle_traindata = shuffle_traindata

        # Datasets will be created in setup()
        self.train_dataset = None
        self.test_dataset = None
        
    def prepare_data(self) -> None:
        """Prepare data (e.g., download). Not needed for manifold-genetics outputs."""
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, and testing.

        In 'split' mode:
            - train_dataset uses train_* paths
            - test_dataset uses test_* paths

        In 'full' mode:
            - Both train and test use train_* paths (same data for both)
        """
        if self.mode == "split":
            # Training dataset from train outputs
            logger.info("Creating training dataset from train outputs")
            self.train_dataset = ManifoldGeneticsDataset(
                pca_path=self.train_pca_path,
                admixture_paths=self.train_admixture_paths,
                labels_path=self.train_labels_path or self.labels_path,
                colormap_path=self.train_colormap_path or self.colormap_path,
                embedding_path=self.train_embedding_path,
                label_column=self.train_label_column or self.label_column,
                geographic_labels_path=self.geographic_labels_path,
            )

            # Test dataset from test outputs
            logger.info("Creating test dataset from test outputs")
            self.test_dataset = ManifoldGeneticsDataset(
                pca_path=self.test_pca_path,
                admixture_paths=self.test_admixture_paths,
                labels_path=self.test_labels_path or self.labels_path,
                colormap_path=self.test_colormap_path or self.colormap_path,
                embedding_path=self.test_embedding_path,
                label_column=self.test_label_column or self.label_column,
                geographic_labels_path=self.geographic_labels_path,
            )

        elif self.mode == "full":
            # Use train data for both train and test
            logger.info("Creating full dataset (same data for train and test)")
            self.train_dataset = ManifoldGeneticsDataset(
                pca_path=self.train_pca_path,
                admixture_paths=self.train_admixture_paths,
                labels_path=self.labels_path,
                colormap_path=self.colormap_path,
                embedding_path=self.train_embedding_path,
                label_column=self.label_column,
                geographic_labels_path=self.geographic_labels_path,
            )
            self.test_dataset = self.train_dataset

        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Use 'split' or 'full'.")
    
    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_traindata,
            collate_fn=self._collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader (same as train for unsupervised methods)."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
    
    @staticmethod
    def _collate_fn(batch):
        """
        Collate function for DataLoader.
        
        Converts list of samples into batched tensors.
        """
        data_samples = [torch.tensor(sample["data"], dtype=torch.float32) for sample in batch]
        data_samples = torch.stack(data_samples)
        metadata = [sample["metadata"] for sample in batch]
        return {"data": data_samples, "metadata": metadata}
