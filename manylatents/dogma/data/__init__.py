"""Sequence datasets for central dogma analysis.

Includes:
    - SequenceDataset: Base dataset for biological sequences
    - SequenceDataModule: LightningDataModule for sequence encoding
    - CentralDogmaDataModule: DataModule for aligned DNA/RNA/Protein sequences
    - TEST_SEQUENCES: Preset test sequences for validation

Future home for:
    - ClinVar variant datasets
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

__all__ = [
    "SequenceDataset",
    "SequenceDataModule",
    "CentralDogmaDataModule",
    "CentralDogmaDataset",
    "TEST_SEQUENCES",
]
