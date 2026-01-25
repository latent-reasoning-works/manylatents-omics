"""
ManifoldGeneticsDataModule - Lightning DataModule for manifold-genetics outputs.

This module provides a PyTorch Lightning DataModule that wraps ManifoldGeneticsDataset
and handles train/val/test splits based on fit vs transform outputs from manifold-genetics.

This replaces the legacy biobank-specific DataModules (HGDPDataModule, AOUDataModule, etc.)
with a single, dataset-agnostic implementation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

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
    - Loading fit (train) and transform (test) outputs from manifold-genetics
    - Creating train/val/test DataLoaders
    - Feature selection (which PCs, which K)
    - Batching and multiprocessing
    
    Train/test splits are determined by manifold-genetics outputs:
    - fit_* files → training data
    - transform_* files → test data
    
    Args:
        fit_pca_path: Path to fit PCA CSV (training samples)
        transform_pca_path: Path to transform PCA CSV (test samples)
        fit_admixture_paths: Dict mapping K to fit admixture paths
        transform_admixture_paths: Dict mapping K to transform admixture paths
        labels_path: Path to labels.csv (must contain all fit+transform samples)
        colormap_path: Path to colormap.json
        fit_embedding_path: Optional path to fit embedding CSV
        transform_embedding_path: Optional path to transform embedding CSV
        label_column: Column name in labels.csv to use (default: "Population")
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        mode: 'split' (separate train/test) or 'full' (same data for both)
        shuffle_traindata: Whether to shuffle training data
        
    Example config (configs/data/manifold_genetics_hgdp.yaml):
        ```yaml
        _target_: manylatents.popgen.data.ManifoldGeneticsDataModule
        batch_size: 128
        num_workers: 4
        mode: split
        
        # Paths to manifold-genetics outputs
        fit_pca_path: ./data/hgdp/manifold_genetics/pca/fit_pca_50.csv
        transform_pca_path: ./data/hgdp/manifold_genetics/pca/transform_pca_50.csv
        
        fit_admixture_paths:
          5: ./data/hgdp/manifold_genetics/admixture/fit.K5.csv
          7: ./data/hgdp/manifold_genetics/admixture/fit.K7.csv
        transform_admixture_paths:
          5: ./data/hgdp/manifold_genetics/admixture/transform.K5.csv
          7: ./data/hgdp/manifold_genetics/admixture/transform.K7.csv
        
        labels_path: ./data/hgdp/manifold_genetics/labels.csv
        colormap_path: ./data/hgdp/manifold_genetics/colormap.json
        label_column: Population
        ```
    """
    
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        fit_pca_path: Optional[str] = None,
        transform_pca_path: Optional[str] = None,
        fit_admixture_paths: Optional[Dict[int, str]] = None,
        transform_admixture_paths: Optional[Dict[int, str]] = None,
        labels_path: Optional[str] = None,
        colormap_path: Optional[str] = None,
        fit_embedding_path: Optional[str] = None,
        transform_embedding_path: Optional[str] = None,
        label_column: str = "Population",
        geographic_labels_path: Optional[str] = None,
        mode: str = "split",
        shuffle_traindata: bool = True,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fit_pca_path = fit_pca_path
        self.transform_pca_path = transform_pca_path
        self.fit_admixture_paths = fit_admixture_paths
        self.transform_admixture_paths = transform_admixture_paths
        self.labels_path = labels_path
        self.colormap_path = colormap_path
        self.fit_embedding_path = fit_embedding_path
        self.transform_embedding_path = transform_embedding_path
        self.label_column = label_column
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
            - train_dataset uses fit_* paths
            - test_dataset uses transform_* paths
            
        In 'full' mode:
            - Both train and test use the same data (transform_* paths, or fit_* if transform not available)
        """
        if self.mode == "split":
            # Training dataset from fit outputs
            logger.info("Creating training dataset from fit outputs")
            self.train_dataset = ManifoldGeneticsDataset(
                pca_path=self.fit_pca_path,
                admixture_paths=self.fit_admixture_paths,
                labels_path=self.labels_path,
                colormap_path=self.colormap_path,
                embedding_path=self.fit_embedding_path,
                label_column=self.label_column,
                geographic_labels_path=self.geographic_labels_path,
            )

            # Test dataset from transform outputs
            logger.info("Creating test dataset from transform outputs")
            self.test_dataset = ManifoldGeneticsDataset(
                pca_path=self.transform_pca_path,
                admixture_paths=self.transform_admixture_paths,
                labels_path=self.labels_path,
                colormap_path=self.colormap_path,
                embedding_path=self.transform_embedding_path,
                label_column=self.label_column,
                geographic_labels_path=self.geographic_labels_path,
            )

        elif self.mode == "full":
            # Use transform data (or fit if transform not available) for both train and test
            pca_path = self.transform_pca_path or self.fit_pca_path
            admixture_paths = self.transform_admixture_paths or self.fit_admixture_paths
            embedding_path = self.transform_embedding_path or self.fit_embedding_path

            logger.info("Creating full dataset (same data for train and test)")
            self.train_dataset = ManifoldGeneticsDataset(
                pca_path=pca_path,
                admixture_paths=admixture_paths,
                labels_path=self.labels_path,
                colormap_path=self.colormap_path,
                embedding_path=embedding_path,
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
