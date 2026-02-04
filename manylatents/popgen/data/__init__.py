"""Population genetics datasets."""

from .plink_dataset import PlinkDataset
from .hgdp_dataset import HGDPDataset
from .hgdp import HGDPDataModule
from .aou_dataset import AOUDataset
from .aou import AOUDataModule
from .ukbb_dataset import UKBBDataset
from .ukbb import UKBBDataModule
from .mhi_dataset import MHIDataset
from .mhi import MHIDataModule

__all__ = [
    # Core
    "PlinkDataset",
    # Genomics (PLINK-based)
    "HGDPDataset",
    "HGDPDataModule",
    "AOUDataset",
    "AOUDataModule",
    "UKBBDataset",
    "UKBBDataModule",
    "MHIDataset",
    "MHIDataModule",
]
