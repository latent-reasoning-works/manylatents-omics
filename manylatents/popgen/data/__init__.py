"""Population genetics datasets."""

# New manifold-genetics based interface (recommended)
from .manifold_genetics_dataset import ManifoldGeneticsDataset
from .manifold_genetics import ManifoldGeneticsDataModule

# Legacy PLINK-based datasets (deprecated - use ManifoldGeneticsDataset instead)
from .plink_dataset import PlinkDataset
from .precomputed_mixin import PrecomputedMixin
from .hgdp_dataset import HGDPDataset
from .hgdp import HGDPDataModule
from .aou_dataset import AOUDataset
from .aou import AOUDataModule
from .ukbb_dataset import UKBBDataset
from .ukbb import UKBBDataModule
from .mhi_dataset import MHIDataset
from .mhi import MHIDataModule

__all__ = [
    # New interface (manifold-genetics based)
    "ManifoldGeneticsDataset",
    "ManifoldGeneticsDataModule",
    # Legacy (PLINK-based) - deprecated
    "PlinkDataset",
    "PrecomputedMixin",
    "HGDPDataset",
    "HGDPDataModule",
    "AOUDataset",
    "AOUDataModule",
    "UKBBDataset",
    "UKBBDataModule",
    "MHIDataset",
    "MHIDataModule",
]
