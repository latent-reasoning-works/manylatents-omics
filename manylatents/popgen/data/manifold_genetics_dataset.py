"""
ManifoldGeneticsDataset - Dataset-agnostic loader for manifold-genetics outputs.

This module provides a PyTorch Dataset that consumes standardized outputs from
the manifold-genetics package, replacing the legacy PLINK-based dataset classes
that contained biobank-specific logic.

Expected manifold-genetics output structure:
    output_dir/
    ├── pca/
    │   ├── fit_pca_*.csv       # PCA coordinates for training samples
    │   └── transform_pca_*.csv # PCA coordinates for test samples
    ├── admixture/
    │   ├── fit.K{k}.csv        # Admixture proportions for training (e.g., fit.K5.csv)
    │   └── transform.K{k}.csv  # Admixture proportions for test
    ├── embeddings/
    │   └── *.csv               # Optional custom embeddings
    ├── labels.csv              # Sample labels with sample_id column
    └── colormap.json           # Label-to-color mapping for visualization

All CSVs are aligned by 'sample_id' column.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ManifoldGeneticsDataset(Dataset):
    """
    PyTorch Dataset for manifold-genetics standardized outputs.
    
    This dataset is completely biobank-agnostic and replaces the legacy
    PLINK-based dataset classes (HGDPDataset, AOUDataset, UKBBDataset, MHIDataset).
    
    All preprocessing, filtering, and subsetting is handled upstream by manifold-genetics.
    This class only handles:
    - Loading CSVs
    - Joining by sample_id
    - Exposing tensors for PyTorch DataLoaders
    
    Args:
        pca_path: Path to PCA coordinates CSV (contains sample_id column)
        admixture_paths: Dict mapping K values to admixture CSV paths (e.g., {5: "fit.K5.csv"})
        labels_path: Path to labels.csv (must contain sample_id and label columns)
        colormap_path: Path to colormap.json for visualization
        embedding_path: Optional path to custom embedding CSV
        label_column: Column name in labels.csv to use as labels (default: "Population")
        
    CSV Format Requirements:
        - All CSVs must have a 'sample_id' column for joining
        - PCA CSV: sample_id, dim_1, dim_2, ..., dim_n
        - Admixture CSV: sample_id, Ancestry1, Ancestry2, ..., AncestryK (or numbered columns)
        - Labels CSV: sample_id, <label_column>, [other metadata columns]
        - Embeddings CSV: sample_id, dim_1, dim_2, ..., dim_n (or custom column names)
    """
    
    def __init__(
        self,
        pca_path: Optional[str] = None,
        admixture_paths: Optional[Dict[int, str]] = None,
        labels_path: Optional[str] = None,
        colormap_path: Optional[str] = None,
        embedding_path: Optional[str] = None,
        label_column: str = "Population",
    ):
        super().__init__()
        
        self.pca_path = pca_path
        self.admixture_paths = admixture_paths or {}
        self.labels_path = labels_path
        self.colormap_path = colormap_path
        self.embedding_path = embedding_path
        self.label_column = label_column
        
        # Load data
        self.data_df = None
        self.labels_df = None
        self.colormap = None
        self.sample_ids = None
        
        self._load_data()
        
    def _load_data(self):
        """Load and merge all data sources by sample_id."""
        dfs_to_merge = []
        
        # Load PCA (columns: sample_id, dim_1, dim_2, ..., dim_n)
        if self.pca_path:
            logger.info(f"Loading PCA from {self.pca_path}")
            pca_df = pd.read_csv(self.pca_path)
            if 'sample_id' not in pca_df.columns:
                raise ValueError(f"PCA CSV must contain 'sample_id' column. Found: {pca_df.columns.tolist()}")
            dfs_to_merge.append(pca_df)
            # Count dimensions (columns that start with 'dim_' or all columns except sample_id)
            dim_cols = [c for c in pca_df.columns if c.startswith('dim_') or (c != 'sample_id' and c not in ['latitude', 'longitude'])]
            logger.info(f"Loaded {len(pca_df)} samples with {len(dim_cols)} PCA dimensions (columns: dim_1, dim_2, ...)")
        
        # Load Admixture
        if self.admixture_paths:
            for k, admix_path in self.admixture_paths.items():
                logger.info(f"Loading admixture K={k} from {admix_path}")
                admix_df = pd.read_csv(admix_path)
                if 'sample_id' not in admix_df.columns:
                    raise ValueError(f"Admixture CSV must contain 'sample_id' column. Found: {admix_df.columns.tolist()}")
                # Rename columns to include K value to avoid conflicts
                admix_df = admix_df.rename(columns={
                    col: f"K{k}_{col}" for col in admix_df.columns if col != 'sample_id'
                })
                dfs_to_merge.append(admix_df)
                logger.info(f"Loaded {len(admix_df)} samples with K={k} ancestries")
        
        # Load custom embedding
        if self.embedding_path:
            logger.info(f"Loading embedding from {self.embedding_path}")
            embed_df = pd.read_csv(self.embedding_path)
            if 'sample_id' not in embed_df.columns:
                raise ValueError(f"Embedding CSV must contain 'sample_id' column. Found: {embed_df.columns.tolist()}")
            dfs_to_merge.append(embed_df)
            logger.info(f"Loaded {len(embed_df)} samples with {len(embed_df.columns)-1} embedding dimensions")
        
        # Merge all data sources
        if not dfs_to_merge:
            raise ValueError("No data sources provided. Must specify at least one of: pca_path, admixture_paths, embedding_path")
        
        self.data_df = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            self.data_df = self.data_df.merge(df, on='sample_id', how='inner')
            logger.info(f"After merge: {len(self.data_df)} samples remaining")
        
        # Load labels
        if self.labels_path:
            logger.info(f"Loading labels from {self.labels_path}")
            self.labels_df = pd.read_csv(self.labels_path)
            if 'sample_id' not in self.labels_df.columns:
                raise ValueError(f"Labels CSV must contain 'sample_id' column. Found: {self.labels_df.columns.tolist()}")
            if self.label_column not in self.labels_df.columns:
                raise ValueError(
                    f"Label column '{self.label_column}' not found in labels CSV. "
                    f"Available columns: {self.labels_df.columns.tolist()}"
                )
            
            # Merge with data
            self.data_df = self.data_df.merge(self.labels_df, on='sample_id', how='inner')
            logger.info(f"After merging labels: {len(self.data_df)} samples with labels")
        
        # Load colormap
        if self.colormap_path:
            logger.info(f"Loading colormap from {self.colormap_path}")
            with open(self.colormap_path, 'r') as f:
                self.colormap = json.load(f)
            logger.info(f"Loaded colormap with {len(self.colormap)} entries")
        
        # Extract sample IDs
        self.sample_ids = self.data_df['sample_id'].values
        
        # Separate data columns from metadata
        data_columns = [col for col in self.data_df.columns if col != 'sample_id']
        if self.labels_df is not None:
            # Exclude label and metadata columns from data tensor
            label_cols = set(self.labels_df.columns) - {'sample_id'}
            data_columns = [col for col in data_columns if col not in label_cols]
        
        self.data_array = self.data_df[data_columns].values
        logger.info(f"Final dataset: {len(self)} samples × {self.data_array.shape[1]} features")
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            dict: {
                'data': np.ndarray of features (PCA/admixture/embeddings),
                'metadata': dict of sample metadata including labels
            }
        """
        sample_data = self.data_array[idx]
        
        # Get all metadata for this sample
        metadata = self.data_df.iloc[idx].to_dict()
        
        return {
            'data': sample_data,
            'metadata': metadata
        }
    
    def get_labels(self, label_col: Optional[str] = None) -> np.ndarray:
        """
        Get labels for all samples.
        
        Args:
            label_col: Label column to extract (default: self.label_column)
            
        Returns:
            np.ndarray: Array of labels
        """
        if self.labels_df is None:
            raise ValueError("No labels loaded. Provide labels_path to constructor.")
        
        label_col = label_col or self.label_column
        if label_col not in self.data_df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. "
                f"Available: {[c for c in self.data_df.columns if c in self.labels_df.columns]}"
            )
        
        return self.data_df[label_col].values
    
    def get_sample_ids(self) -> np.ndarray:
        """Get sample IDs for all samples."""
        return self.sample_ids
    
    def get_colormap(self) -> Optional[Dict[str, str]]:
        """Get colormap for visualization."""
        return self.colormap
    
    @property
    def latitude(self) -> Optional[pd.Series]:
        """Get latitude coordinates if available in labels."""
        if self.labels_df is not None and 'latitude' in self.data_df.columns:
            return self.data_df['latitude']
        return None
    
    @property
    def longitude(self) -> Optional[pd.Series]:
        """Get longitude coordinates if available in labels."""
        if self.labels_df is not None and 'longitude' in self.data_df.columns:
            return self.data_df['longitude']
        return None
