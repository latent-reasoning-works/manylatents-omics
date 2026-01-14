"""Sequence datasets for central dogma analysis.

Includes:
    - SequenceDataset: Base dataset for biological sequences
    - SequenceDataModule: LightningDataModule for sequence encoding
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

__all__ = [
    "SequenceDataset",
    "SequenceDataModule",
    "TEST_SEQUENCES",
]
