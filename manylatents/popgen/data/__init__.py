"""Population genetics datasets."""

from .manifold_genetics_dataset import ManifoldGeneticsDataset
from .manifold_genetics import ManifoldGeneticsDataModule

__all__ = [
    "ManifoldGeneticsDataset",
    "ManifoldGeneticsDataModule",
]
