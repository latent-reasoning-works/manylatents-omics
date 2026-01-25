"""DataModule for ClinVar variant sequences.

Loads preprocessed ClinVar data (from download_clinvar.py) and provides
batched sequences for foundation model encoding.

Key difference from CentralDogmaDataModule:
- CentralDogma: get_sequences() → Dict[str, str] (single sequence)
- ClinVar: get_sequences() → Dict[str, List[str]] (batch of sequences)

Example:
    >>> from manylatents.dogma.data import ClinVarDataModule
    >>> dm = ClinVarDataModule(data_dir="data/clinvar")
    >>> dm.setup()
    >>> sequences = dm.get_sequences()
    >>> print(len(sequences["dna"]))  # Number of variants
    >>> labels = dm.get_labels()  # 0=benign, 1=pathogenic
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def parse_fasta(fasta_path: Path) -> Dict[str, str]:
    """Parse FASTA file into dict of id -> sequence."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]  # Take first word after >
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget last sequence
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)

    return sequences


class ClinVarDataset(Dataset):
    """Dataset containing ClinVar variant sequences.

    Each item contains sequences for all three modalities (DNA, RNA, Protein)
    plus metadata (variant_id, label).
    """

    def __init__(
        self,
        variant_ids: List[str],
        dna_sequences: Dict[str, str],
        rna_sequences: Dict[str, str],
        protein_sequences: Dict[str, str],
        labels: np.ndarray,
        metadata: Optional[Dict[str, List]] = None,
    ):
        """
        Args:
            variant_ids: List of variant IDs (e.g., "clinvar_12345")
            dna_sequences: Dict mapping variant_id to DNA sequence
            rna_sequences: Dict mapping variant_id to RNA sequence
            protein_sequences: Dict mapping variant_id to protein sequence
            labels: Array of labels (0=benign, 1=pathogenic)
            metadata: Optional additional metadata per variant
        """
        self.variant_ids = variant_ids
        self.dna_sequences = dna_sequences
        self.rna_sequences = rna_sequences
        self.protein_sequences = protein_sequences
        self.labels = labels
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.variant_ids)

    def __getitem__(self, idx: int) -> Dict:
        var_id = self.variant_ids[idx]
        return {
            "variant_id": var_id,
            "dna": self.dna_sequences.get(var_id, ""),
            "rna": self.rna_sequences.get(var_id, ""),
            "protein": self.protein_sequences.get(var_id, ""),
            "label": self.labels[idx],
        }


class ClinVarDataModule(LightningDataModule):
    """DataModule for ClinVar variant sequences.

    Loads preprocessed ClinVar data and provides batched sequences for
    foundation model encoding. Supports filtering by gene, pathogenicity,
    and maximum variant count.

    Args:
        data_dir: Path to ClinVar data directory (from download_clinvar.py)
        genes: Filter to specific gene symbols (None = all)
        pathogenicity: Filter by label - 'pathogenic', 'benign', 'all'
        max_variants: Maximum number of variants to load (None = all)
        batch_size: Batch size for dataloaders

    Example:
        >>> dm = ClinVarDataModule(data_dir="data/clinvar", genes=["BRCA1"])
        >>> dm.setup()
        >>> seqs = dm.get_sequences()
        >>> print(len(seqs["dna"]))  # Number of BRCA1 variants
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        genes: Optional[List[str]] = None,
        pathogenicity: str = "all",
        max_variants: Optional[int] = None,
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.genes = [g.upper() for g in genes] if genes else None
        self.pathogenicity = pathogenicity
        self.max_variants = max_variants
        self.batch_size = batch_size

        # Will be populated in setup()
        self._variant_ids: List[str] = []
        self._dna_sequences: Dict[str, str] = {}
        self._rna_sequences: Dict[str, str] = {}
        self._protein_sequences: Dict[str, str] = {}
        self._labels: np.ndarray = np.array([])
        self._metadata: Dict[str, List] = {}

        self.dataset: Optional[ClinVarDataset] = None

    def _load_variants_tsv(self) -> None:
        """Load and filter variants from TSV file."""
        tsv_path = self.data_dir / "variants.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(
                f"variants.tsv not found in {self.data_dir}. "
                f"Run scripts/download_clinvar.py first."
            )

        variant_ids = []
        labels = []
        metadata = {
            "gene_symbol": [],
            "clinical_significance": [],
            "review_status": [],
            "chromosome": [],
            "start": [],
            "stop": [],
            "variant_type": [],
        }

        with open(tsv_path) as f:
            header = f.readline().strip().split("\t")
            col_idx = {name: i for i, name in enumerate(header)}

            for line in f:
                fields = line.strip().split("\t")

                # Filter by gene
                gene = fields[col_idx["gene_symbol"]].upper()
                if self.genes and gene not in self.genes:
                    continue

                # Filter by pathogenicity
                label = int(fields[col_idx["label"]])
                if self.pathogenicity == "pathogenic" and label != 1:
                    continue
                elif self.pathogenicity == "benign" and label != 0:
                    continue
                elif self.pathogenicity == "all" and label == -1:
                    continue  # Skip VUS even in "all" mode

                # Max variants limit
                if self.max_variants and len(variant_ids) >= self.max_variants:
                    break

                var_id = f"clinvar_{fields[col_idx['variation_id']]}"
                variant_ids.append(var_id)
                labels.append(label)

                # Collect metadata
                metadata["gene_symbol"].append(fields[col_idx["gene_symbol"]])
                metadata["clinical_significance"].append(
                    fields[col_idx["clinical_significance"]]
                )
                metadata["review_status"].append(fields[col_idx["review_status"]])
                metadata["chromosome"].append(fields[col_idx["chromosome"]])
                metadata["start"].append(int(fields[col_idx["start"]]))
                metadata["stop"].append(int(fields[col_idx["stop"]]))
                metadata["variant_type"].append(fields[col_idx["variant_type"]])

        self._variant_ids = variant_ids
        self._labels = np.array(labels, dtype=np.int64)
        self._metadata = metadata

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data from files."""
        # Load variant metadata first (for filtering)
        self._load_variants_tsv()

        if len(self._variant_ids) == 0:
            raise ValueError(
                f"No variants found after filtering. "
                f"genes={self.genes}, pathogenicity={self.pathogenicity}"
            )

        # Load sequence files
        dna_path = self.data_dir / "dna.fasta"
        rna_path = self.data_dir / "rna.fasta"
        protein_path = self.data_dir / "protein.fasta"

        if dna_path.exists():
            self._dna_sequences = parse_fasta(dna_path)
        if rna_path.exists():
            self._rna_sequences = parse_fasta(rna_path)
        if protein_path.exists():
            self._protein_sequences = parse_fasta(protein_path)

        # Create dataset
        self.dataset = ClinVarDataset(
            variant_ids=self._variant_ids,
            dna_sequences=self._dna_sequences,
            rna_sequences=self._rna_sequences,
            protein_sequences=self._protein_sequences,
            labels=self._labels,
            metadata=self._metadata,
        )

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict:
        """Collate batch of items into batched dict."""
        return {
            "variant_ids": [item["variant_id"] for item in batch],
            "dna": [item["dna"] for item in batch],
            "rna": [item["rna"] for item in batch],
            "protein": [item["protein"] for item in batch],
            "labels": torch.tensor([item["label"] for item in batch]),
        }

    def get_sequences(self) -> Dict[str, List[str]]:
        """Return dict with lists of sequences for each modality.

        This is the primary interface for batch encoding.

        Returns:
            Dict with keys 'dna', 'rna', 'protein' mapping to lists of sequences.
            Sequences are ordered by variant_id.
        """
        if self.dataset is None:
            self.setup()

        return {
            "dna": [self._dna_sequences.get(v, "") for v in self._variant_ids],
            "rna": [self._rna_sequences.get(v, "") for v in self._variant_ids],
            "protein": [self._protein_sequences.get(v, "") for v in self._variant_ids],
        }

    def get_variant_ids(self) -> List[str]:
        """Return list of variant IDs in order."""
        if self.dataset is None:
            self.setup()
        return self._variant_ids

    def get_labels(self) -> np.ndarray:
        """Return pathogenicity labels (0=benign, 1=pathogenic)."""
        if self.dataset is None:
            self.setup()
        return self._labels

    def get_metadata(self) -> Dict[str, List]:
        """Return metadata dict with lists aligned to variant_ids."""
        if self.dataset is None:
            self.setup()
        return self._metadata

    @property
    def num_variants(self) -> int:
        """Number of variants loaded."""
        return len(self._variant_ids)

    @property
    def num_pathogenic(self) -> int:
        """Number of pathogenic variants."""
        return int((self._labels == 1).sum())

    @property
    def num_benign(self) -> int:
        """Number of benign variants."""
        return int((self._labels == 0).sum())

    def __repr__(self) -> str:
        if self.dataset is None:
            return f"ClinVarDataModule(data_dir='{self.data_dir}', not_setup=True)"
        return (
            f"ClinVarDataModule("
            f"n_variants={self.num_variants}, "
            f"n_pathogenic={self.num_pathogenic}, "
            f"n_benign={self.num_benign}, "
            f"genes={self.genes})"
        )
