"""Sequence datasets for central dogma analysis.

Includes:
    - SequenceDataset: Base dataset for biological sequences
    - SequenceDataModule: LightningDataModule for sequence encoding
    - CentralDogmaDataModule: DataModule for aligned DNA/RNA/Protein sequences
    - ClinVarDataModule: DataModule for ClinVar variant sequences
    - VariantTableDataModule: a chrom/pos/ref/alt table whose datasets yield row indices
    - TEST_SEQUENCES: Preset test sequences for validation

Future home for:
    - ProteinGym benchmarks
    - gnomAD population data
"""

from .sequence_dataset import (
    SequenceDataset,
    SequenceDataModule,
    TEST_SEQUENCES,
)
from .central_dogma_dataset import (
    CentralDogmaDataModule,
    CentralDogmaDataset,
)
from .clinvar_dataset import (
    ClinVarDataModule,
    ClinVarDataset,
)
from .variant_datamodule import VariantDataModule
from .variant_table import VariantTableDataModule

__all__ = [
    "SequenceDataset",
    "SequenceDataModule",
    "CentralDogmaDataModule",
    "CentralDogmaDataset",
    "ClinVarDataModule",
    "ClinVarDataset",
    "VariantDataModule",
    "VariantTableDataModule",
    "TEST_SEQUENCES",
]
