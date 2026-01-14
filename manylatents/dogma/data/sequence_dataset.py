"""
Sequence datasets for foundation model encoders.

Provides sequences (DNA, RNA, protein) as input for FoundationEncoders,
integrating with manylatents DataModule pattern.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset

# Test sequences representing the same biological information
# Each preset contains DNA, RNA, and protein forms of the same sequence
TEST_SEQUENCES = {
    "synthetic_8aa": {
        "dna": "ATGGCTTGGAAACGTGCTCAGGCTTGA",
        "rna": "AUGGCUUGGAAACGUGCUCAGGCUUGA",
        "protein": "MAWKRAQA",
        "description": "Synthetic 8 amino acid test sequence",
    },
    "gfp": {
        # GFP coding sequence - 238 aa
        "dna": "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA",
        # RNA: DNA with T->U substitution (mature mRNA transcript)
        "rna": "AUGGUGAGCAAGGGCGAGGAGCUGUUCACCGGGGUGGUGCCCAUCCUGGUCGAGCUGGACGGCGACGUAAACGGCCACAAGUUCAGCGUGUCCGGCGAGGGCGAGGGCGAUGCCACCUACGGCAAGCUGACCCUGAAGUUCAUCUGCACCACCGGCAAGCUGCCCGUGCCCUGGCCCACCCUCGUGACCACCCUGACCUACGGCGUGCAGUGCUUCAGCCGCUACCCCGACCACAUGAAGCAGCACGACUUCUUCAAGUCCGCCAUGCCCGAAGGCUACGUCCAGGAGCGCACCAUCUUCUUCAAGGACGACGGCAACUACAAGACCCGCGCCGAGGUGAAGUUCGAGGGCGACACCCUGGUGAACCGCAUCGAGCUGAAGGGCAUCGACUUCAAGGAGGACGGCAACAUCCUGGGGCACAAGCUGGAGUACAACUACAACAGCCACAACGUCUAUAUCAUGGCCGACAAGCAGAAGAACGGCAUCAAGGUGAACUUCAAGAUCCGCCACAACAUCGAGGACGGCAGCGUGCAGCUCGCCGACCACUACCAGCAGAACACCCCCCAUCGGCGACGGCCCCGUGCUGCUGCCCGACAACCACUACCUGAGCACCCAGUCCGCCCUGAGCAAAGACCCCAACGAGAAGCGCGAUCACAUGGUCCUGCUGGAGUUCGUGACCGCCGCCGGGAUCACUCUCGGCAUGGACGAGCUGUACAAGUAA",
        "protein": "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
        "description": "Green Fluorescent Protein (GFP) coding sequence - 238 aa",
    },
    "brca1": {
        # BRCA1 exon 11 fragment (first 500bp) - pathogenic variant testing
        "dna": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAAAGGAGCCTACAAGAAAGTACGAGATTTAGTCAACTTGTTGAAGAGCTATTGAAAATCATTTGTGCTTTTCAGCTTGACACAGGTTTGGAGTATGCAAACAGCTATAATTTTGCAAAAAAGGAAAATAACTCTCCTGAACATCTAAAAGATGAAGTTTCTATCATCCAAAGTATGGGCTACAGAAACCGTGCCAAAAGACTTCTACAGAGTGAACCCGAAAATCCTTCCTTGCAGGAAACCAGTCTCAGTGTCCAACTCTCTAACCTTGGAACTGTGAGAACTCTGAGGACAAAGCAGCGGATACAACCTCAAAAGACGTCTGTCTACATTGAATTGGGATCTGATT",
        # RNA: DNA with T->U substitution (mature mRNA transcript)
        "rna": "AUGGAUUUAUCUGCUCUUCGCGUUGAAGAAGUACAAAAUGUCAUUAAUGCUAUGCAGAAAAUCUUAGAGUGUCCCAUCUGUCUGGAGUUGAUCAAGGAACCUGUCUCCACAAAGUGUGACCACAUAUUUUGCAAAUUUUGCAUGCUGAAACUUCUCAACCAGAAGAAAGGGCCUUCACAGUGUCCUUUAUGUAAGAAUGAUAUAACCAAAAGGAGCCUACAAGAAAGUACGAGAUUUAGUCAACUUGUUGAAGAGCUAUUGAAAAUCAUUUGUGCUUUUCAGCUUGACACAGGUUUGGAGUAUGCAAACAGCUAUAAUUUUGCAAAAAAGGAAAAUAACUCUCCUGAACAUCUAAAAGAUGAAGUUUCUAUCAUCCAAAGUAUGGGCUACAGAAACCGUGCCAAAAGACUUCUACAGAGUGAACCCGAAAAUCCUUCCUUGCAGGAAACCAGUCUCAGUGUCCAACUCUCUAACCUUGGAACUGUGAGAACUCUGAGGACAAAGCAGCGGAUACAACCUCAAAAGACGUCUGUCUACAUUGAAUUGGGAUCUGAUU",
        "protein": "MDLSALRREEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKKLIQSEPIGIKAQITVNVKRIHEFESMPKGRIALLDEVLNAVHKNCQKMTEGYQYSGSDLLQQSARQKLQNLLSNVKRQNELENVKNEISSLISNPQKMFQGVQKQLILAIQNLGKYTGPSSSDLKQLNAINKIQNLSNALQLQIKSNANFKKD",
        "description": "BRCA1 exon 11 fragment - pathogenic variant hotspot",
    },
}


class SequenceDataset(Dataset):
    """Dataset containing biological sequences for encoding."""

    def __init__(
        self,
        sequences: List[str],
        modality: str,
        names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            sequences: List of sequences (DNA, RNA, or protein).
            modality: Type of sequences ('dna', 'rna', or 'protein').
            names: Optional names for each sequence.
            metadata: Optional metadata dict.
        """
        self.sequences = sequences
        self.modality = modality
        self.names = names or [f"seq_{i}" for i in range(len(sequences))]
        self.metadata = metadata or {}

        # For compatibility with manylatents metrics
        # Store sequences as 'data' attribute (as character indices for now)
        self._char_to_idx = self._build_vocab()
        self.data = self._encode_sequences()

    def _build_vocab(self) -> Dict[str, int]:
        """Build character vocabulary based on modality."""
        if self.modality == "dna":
            chars = "ACGTN"
        elif self.modality == "rna":
            chars = "ACGUN"
        elif self.modality == "protein":
            chars = "ACDEFGHIKLMNPQRSTVWY"
        else:
            # Build from sequences
            chars = "".join(set("".join(self.sequences)))
        return {c: i for i, c in enumerate(chars)}

    def _encode_sequences(self) -> np.ndarray:
        """Encode sequences as integer arrays for metrics compatibility."""
        max_len = max(len(s) for s in self.sequences)
        encoded = np.zeros((len(self.sequences), max_len), dtype=np.int64)
        for i, seq in enumerate(self.sequences):
            for j, c in enumerate(seq):
                encoded[i, j] = self._char_to_idx.get(c, 0)
        return encoded

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "data": torch.from_numpy(self.data[idx]).float(),  # Encoded sequence tensor (for pipeline compat)
            "sequence": self.sequences[idx],  # Raw string for encoder
            "name": self.names[idx],
            "modality": self.modality,
        }


class SequenceDataModule(LightningDataModule):
    """DataModule for sequence datasets.

    Provides sequences in a format compatible with manylatents while
    supporting FoundationEncoder's string input requirements.
    """

    def __init__(
        self,
        sequences: Optional[List[str]] = None,
        modality: str = "protein",
        preset: str = "synthetic_short",
        batch_size: int = 1,
        mode: str = "full",
    ):
        """
        Args:
            sequences: Custom sequences. If None, uses preset.
            modality: Sequence type ('dna', 'rna', 'protein').
            preset: Preset sequence set name (if sequences is None).
            batch_size: Batch size for dataloader.
            mode: 'full' or 'split' for train/test splitting.
        """
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.mode = mode
        self.modality = modality

        # Get sequences from preset or use provided
        if sequences is None:
            preset_data = TEST_SEQUENCES.get(preset, TEST_SEQUENCES["synthetic_8aa"])
            if modality in preset_data:
                self.sequences = [preset_data[modality]]
                self.names = [preset]
            else:
                raise ValueError(f"No {modality} sequence in preset '{preset}'")
        else:
            self.sequences = sequences
            self.names = [f"seq_{i}" for i in range(len(sequences))]

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        self.train_dataset = SequenceDataset(
            sequences=self.sequences,
            modality=self.modality,
            names=self.names,
        )
        self.test_dataset = self.train_dataset  # Same for inference-only

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def get_sequences(self) -> List[str]:
        """Return raw sequences for encoder input."""
        return self.sequences

    def get_tensor(self) -> torch.Tensor:
        """Return encoded sequences as tensor (for LatentModule compatibility)."""
        if self.train_dataset is None:
            self.setup()
        return torch.from_numpy(self.train_dataset.data).float()
