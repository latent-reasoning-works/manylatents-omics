"""DataModule providing aligned DNA/RNA/Protein sequences.

Central Dogma DataModule provides sequences for all three modalities
(DNA, RNA, Protein) aligned to represent the same biological information.
This enables fusion experiments with CentralDogmaFusion.

Example:
    >>> from manylatents.dogma.data import CentralDogmaDataModule
    >>> dm = CentralDogmaDataModule(preset="gfp")
    >>> dm.setup()
    >>> sequences = dm.get_sequences()
    >>> print(sequences.keys())  # dict_keys(['dna', 'rna', 'protein'])
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .sequence_dataset import TEST_SEQUENCES


class CentralDogmaDataset(Dataset):
    """Dataset containing aligned DNA/RNA/Protein sequences.

    Each item contains all three modalities for the same biological sequence.
    """

    def __init__(
        self,
        sequences: Dict[str, str],
        name: str = "unnamed",
    ):
        """
        Args:
            sequences: Dict with 'dna', 'rna', 'protein' keys.
            name: Identifier for this sequence set.
        """
        self.sequences = sequences
        self.name = name

        # For manylatents compatibility, create a data tensor
        # We'll use protein length as the representative size
        self._char_to_idx = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        protein = sequences["protein"]
        encoded = np.zeros(len(protein), dtype=np.int64)
        for i, c in enumerate(protein):
            encoded[i] = self._char_to_idx.get(c, 0)
        self.data = encoded

    def __len__(self) -> int:
        return 1  # Single aligned sequence set

    def __getitem__(self, idx: int) -> Dict:
        return {
            "data": torch.from_numpy(self.data).float(),
            "sequences": self.sequences,
            "name": self.name,
        }


class CentralDogmaDataModule(LightningDataModule):
    """DataModule providing sequences for all 3 central dogma modalities.

    Provides aligned DNA, RNA, and Protein sequences from preset test
    sequences or custom inputs. Designed for use with CentralDogmaFusion.

    Args:
        preset: Name of preset sequence set ('synthetic_8aa', 'gfp', 'brca1').
        sequences: Custom sequences dict with 'dna', 'rna', 'protein' keys.
                   If provided, overrides preset.
        batch_size: Batch size for dataloaders.

    Example:
        >>> dm = CentralDogmaDataModule(preset="gfp")
        >>> dm.setup()
        >>> seqs = dm.get_sequences()
        >>> print(len(seqs["protein"]))  # 238 for GFP
    """

    def __init__(
        self,
        preset: str = "gfp",
        sequences: Optional[Dict[str, str]] = None,
        batch_size: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.preset = preset
        self.batch_size = batch_size

        # Use custom sequences or load from preset
        if sequences is not None:
            self._validate_sequences(sequences)
            self._sequences = sequences
            self._name = "custom"
        else:
            self._sequences = None
            self._name = preset

        self.train_dataset = None
        self.test_dataset = None

    def _validate_sequences(self, sequences: Dict[str, str]) -> None:
        """Validate that sequences dict has required keys."""
        required_keys = {"dna", "rna", "protein"}
        missing = required_keys - set(sequences.keys())
        if missing:
            raise ValueError(
                f"Sequences dict missing required keys: {missing}. "
                f"Expected: {required_keys}"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets."""
        if self._sequences is None:
            # Load from preset
            if self.preset not in TEST_SEQUENCES:
                available = list(TEST_SEQUENCES.keys())
                raise ValueError(
                    f"Unknown preset '{self.preset}'. Available: {available}"
                )
            preset_data = TEST_SEQUENCES[self.preset]
            self._sequences = {
                "dna": preset_data["dna"],
                "rna": preset_data["rna"],
                "protein": preset_data["protein"],
            }

        self.train_dataset = CentralDogmaDataset(
            sequences=self._sequences,
            name=self._name,
        )
        self.test_dataset = self.train_dataset  # Same for inference-only

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup()
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def get_sequences(self) -> Dict[str, str]:
        """Return dict with dna, rna, protein sequence strings.

        This is the primary interface for CentralDogmaFusion to get
        sequences for encoding.

        Returns:
            Dict with keys 'dna', 'rna', 'protein' mapping to sequence strings.
        """
        if self._sequences is None:
            self.setup()
        return self._sequences

    def get_tensor(self) -> torch.Tensor:
        """Return encoded protein sequence as tensor (for LatentModule compatibility)."""
        if self.train_dataset is None:
            self.setup()
        return torch.from_numpy(self.train_dataset.data).float().unsqueeze(0)

    @property
    def sequence_lengths(self) -> Dict[str, int]:
        """Return lengths of sequences in each modality."""
        if self._sequences is None:
            self.setup()
        return {
            "dna": len(self._sequences["dna"]),
            "rna": len(self._sequences["rna"]),
            "protein": len(self._sequences["protein"]),
        }

    def __repr__(self) -> str:
        if self._sequences is None:
            return f"CentralDogmaDataModule(preset='{self.preset}', not_setup=True)"
        lengths = self.sequence_lengths
        return (
            f"CentralDogmaDataModule(preset='{self.preset}', "
            f"dna_len={lengths['dna']}, "
            f"rna_len={lengths['rna']}, "
            f"protein_len={lengths['protein']})"
        )
