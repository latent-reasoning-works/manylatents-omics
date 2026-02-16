"""PyTorch Dataset for AnnData objects (single-cell omics data)."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AnnDataset(Dataset):
    """
    PyTorch Dataset for AnnData objects.

    Supports single-cell RNA-seq, ATAC-seq, and other omics data
    stored in the AnnData (.h5ad) format.
    """

    def __init__(
        self,
        adata_path: str,
        label_key: Optional[str] = None,
        layer: Optional[str] = None,
        use_raw: bool = False,
    ):
        """
        Initialize the AnnDataset.

        Parameters
        ----------
        adata_path : str
            Path to the .h5ad file containing the AnnData object.
        label_key : str or None
            Key in adata.obs used as cell labels (e.g., cell type or condition).
            If None, assigns all-zero dummy labels.
        layer : str or None
            If specified, use adata.layers[layer] instead of adata.X.
        use_raw : bool
            If True, use adata.raw.X instead of adata.X.
        """
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError(
                "scanpy is required for AnnDataset. Install with: pip install scanpy"
            )

        adata_path = Path(adata_path)
        if not adata_path.exists():
            raise FileNotFoundError(f"AnnData file not found: {adata_path}")

        logger.info(f"Loading AnnData from: {adata_path}")
        adata = sc.read_h5ad(adata_path)

        # Extract expression matrix
        if use_raw and adata.raw is not None:
            X = adata.raw.X
        elif layer is not None and layer in adata.layers:
            X = adata.layers[layer]
        else:
            X = adata.X

        # Convert sparse to dense if needed
        if hasattr(X, "toarray"):
            X = X.toarray()

        self.data = torch.tensor(X, dtype=torch.float32)
        self.n_samples, self.n_features = self.data.shape

        # Store obs for metadata access
        self._obs = adata.obs.copy()

        # Extract labels — encode as integers for PyTorch collation,
        # keep string names for get_labels() (used by plot/metric callbacks).
        if label_key is not None and label_key in adata.obs:
            cats = adata.obs[label_key].astype("category")
            self.label_names = cats.cat.categories.tolist()
            self.metadata = cats.cat.codes.values.astype(np.int64).copy()
        else:
            self.label_names = None
            self.metadata = np.zeros(self.n_samples, dtype=np.int64)

        # Store var names for reference
        self.feature_names = adata.var_names.tolist() if hasattr(adata, "var_names") else None

        logger.info(f"Loaded {self.n_samples} samples x {self.n_features} features")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "data": self.data[idx],
            "metadata": torch.tensor(self.metadata[idx], dtype=torch.long),
        }

    def get_labels(self) -> np.ndarray:
        """Return cell/sample labels as string names (for plotting/metrics)."""
        if self.label_names is not None:
            return np.array(self.label_names)[self.metadata]
        return self.metadata

    def get_data(self) -> torch.Tensor:
        """Return the full data matrix."""
        return self.data

    def get_obs(self, key: str) -> np.ndarray:
        """Get a specific observation annotation."""
        if key in self._obs.columns:
            return self._obs[key].values
        raise KeyError(f"Key '{key}' not found in obs. Available: {list(self._obs.columns)}")
