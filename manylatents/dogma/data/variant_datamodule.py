"""DataModule for paired WT/MUT variant sequences.

Handles variant-type-specific loading (synonymous, splice_extended, utr_combined)
with paired sequence generation for delta embedding computation.

Key difference from ClinVarDataModule:
- ClinVar: Loads pre-generated sequences from single FASTA per modality
- Variant: Loads WT sequences and injects variants to generate MUT on-the-fly

Usage:
    >>> from manylatents.dogma.data import VariantDataModule
    >>> dm = VariantDataModule(
    ...     variants_dir="data/clinvar/variants",
    ...     sequences_dir="data/clinvar/sequences",
    ...     variant_type="synonymous",
    ...     modality="dna",
    ... )
    >>> dm.setup()
    >>> pairs = dm.get_sequence_pairs()
    >>> print(len(pairs["wt"]), len(pairs["mut"]))

Hydra config:
    python -m manylatents.omics.main data=clinvar_variants \\
        data.variant_type=synonymous data.modality=dna
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def load_fasta(fasta_path: Path) -> Dict[str, str]:
    """Load FASTA file into {id: sequence} dict."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = "".join(current_seq)

    return sequences


def inject_variant_at_position(seq: str, pos: int, ref: str, alt: str) -> str:
    """Inject variant at a known position.

    Args:
        seq: Wild-type sequence
        pos: Position to inject variant (0-indexed)
        ref: Reference allele
        alt: Alternate allele

    Returns:
        Mutant sequence with variant injected
    """
    # Handle RNA (T→U conversion) - alt should match sequence alphabet
    if seq and ("U" in seq or "u" in seq):
        alt = alt.replace("T", "U").replace("t", "u")
    return seq[:pos] + alt + seq[pos + len(ref):]


_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def find_variant_in_rna(
    rna_seq: str, dna_seq: str, ref: str, context_size: int = 15
) -> Optional[tuple]:
    """Find the variant position in gene-level mRNA using DNA context.

    The DNA FASTA is centered on the variant (±8192bp) on the plus strand.
    The RNA is gene-level mature mRNA (always 5'→3' of the gene). For genes
    on the minus strand, the DNA context must be reverse-complemented.

    Args:
        rna_seq: Gene-level mRNA sequence (cDNA alphabet, uses T not U)
        dna_seq: DNA sequence centered on variant (±context, plus strand)
        ref: Reference allele (DNA, plus strand)
        context_size: Number of flanking bases on each side

    Returns:
        Tuple of (position, is_minus_strand) or None if not found.
        When is_minus_strand=True, the caller must complement the alt allele.
    """
    dna_center = len(dna_seq) // 2
    rna_upper = rna_seq.upper()

    # Try context windows from large to small. Larger contexts give unique
    # matches; smaller ones handle exon boundaries where the DNA context
    # spans an intron.
    for ctx_size in [context_size, 10, 7, 5]:
        left = max(0, dna_center - ctx_size)
        right = min(len(dna_seq), dna_center + len(ref) + ctx_size)
        dna_context = dna_seq[left:right].upper()
        variant_offset = dna_center - left

        # Forward strand: search DNA context directly in mRNA
        pos = rna_upper.find(dna_context)
        if pos != -1:
            return (pos + variant_offset, False)

        # Minus strand: reverse-complement the DNA context
        rc_context = _revcomp(dna_context)
        pos = rna_upper.find(rc_context)
        if pos != -1:
            # On minus strand, the variant position is mirrored
            rc_offset = len(rc_context) - variant_offset - len(ref)
            return (pos + rc_offset, True)

    return None


class VariantPairDataset(Dataset):
    """Dataset of (wt_seq, mut_seq, label) tuples."""

    def __init__(self, wt_seqs: List[str], mut_seqs: List[str], labels: List[int]):
        self.wt_seqs = wt_seqs
        self.mut_seqs = mut_seqs
        self.labels = labels
        # Required by experiment.py evaluate(): ds.data.shape
        self.data = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.wt_seqs)

    def __getitem__(self, idx):
        return {
            "wt": self.wt_seqs[idx],
            "mut": self.mut_seqs[idx],
            "label": self.labels[idx],
        }


class VariantDataModule(LightningDataModule):
    """DataModule for paired WT/MUT variant sequences.

    Handles:
    - Loading variant TSV metadata for specific variant types
    - Loading WT sequences from FASTA
    - Injecting variants to create MUT sequences
    - Exposing pairs for delta embedding computation

    Args:
        variants_dir: Directory containing variant TSV files (e.g., synonymous.tsv)
        sequences_dir: Directory containing sequence FASTA files
        variant_type: Type of variants to load (synonymous, splice_extended, utr_combined)
        modality: Sequence modality (dna, rna, protein)
        batch_size: Batch size for dataloaders
        max_variants: Maximum number of variants to load (None = all)
        max_seq_length: Maximum sequence length to include (None = all).
            Sequences longer than this are skipped. Recommended: 1024 for ESM3,
            8192 for Evo2, 4096 for Orthrus.
        valid_variants_file: Path to file with valid variant IDs (one per line).
            If provided, only variants in this file are loaded. This ensures
            consistent variant sets across modalities. Typically generated by
            filtering for variants that satisfy all model constraints.
            Example: data/clinvar/variants/synonymous_valid.txt
    """

    def __init__(
        self,
        variants_dir: Union[str, Path],
        sequences_dir: Union[str, Path],
        variant_type: str,
        modality: str,
        batch_size: int = 32,
        max_variants: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        valid_variants_file: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.variants_dir = Path(variants_dir)
        self.sequences_dir = Path(sequences_dir)
        self.variant_type = variant_type
        self.modality = modality
        self.batch_size = batch_size
        self.max_variants = max_variants
        self.max_seq_length = max_seq_length
        self.valid_variants_file = Path(valid_variants_file) if valid_variants_file else None

        self._wt_seqs: List[str] = []
        self._mut_seqs: List[str] = []
        self._labels: List[int] = []
        self._variant_ids: List[str] = []
        self._metadata: Dict[str, List] = {}

        # Required by manylatents experiment.py (line 215)
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Load variant metadata and sequences."""
        # Load variant TSV
        tsv_path = self.variants_dir / f"{self.variant_type}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(
                f"Variant TSV not found: {tsv_path}. "
                f"Available types: {[f.stem for f in self.variants_dir.glob('*.tsv')]}"
            )

        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
        df['var_id'] = 'clinvar_' + df['variation_id'].astype(str)

        # Filter to valid variants if file provided
        if self.valid_variants_file:
            if not self.valid_variants_file.exists():
                raise FileNotFoundError(
                    f"Valid variants file not found: {self.valid_variants_file}"
                )
            valid_ids = set(pd.read_csv(self.valid_variants_file, header=None)[0].tolist())
            n_before = len(df)
            df = df[df['var_id'].isin(valid_ids)]
            print(f"Filtered to valid variants: {len(df)} / {n_before} "
                  f"(from {self.valid_variants_file.name})")

        if self.max_variants:
            df = df.head(self.max_variants)

        # Load primary modality sequences
        fasta_path = self.sequences_dir / f"{self.variant_type}_{self.modality}.fasta"
        if not fasta_path.exists():
            raise FileNotFoundError(
                f"Sequence FASTA not found: {fasta_path}. "
                f"Run sequence generation script first."
            )

        sequences = load_fasta(fasta_path)

        # For RNA/protein modalities, also load DNA for context-based injection
        dna_sequences = None
        if self.modality in ("rna", "protein"):
            dna_fasta = self.sequences_dir / f"{self.variant_type}_dna.fasta"
            if dna_fasta.exists():
                dna_sequences = load_fasta(dna_fasta)
                print(f"Loaded DNA context for {self.modality} variant injection "
                      f"({len(dna_sequences)} seqs)")
            else:
                print(f"WARNING: DNA FASTA not found at {dna_fasta}. "
                      f"RNA/protein injection will use center heuristic.")

        # Build paired sequences
        skipped_length = 0
        skipped_injection = 0
        for _, row in df.iterrows():
            var_id = row['var_id']
            if var_id not in sequences:
                continue

            wt_seq = sequences[var_id]

            # Skip sequences exceeding max length
            if self.max_seq_length and len(wt_seq) > self.max_seq_length:
                skipped_length += 1
                continue

            ref, alt = str(row["ref"]), str(row["alt"])

            # For protein modality, synonymous variants don't change AA
            if self.modality == "protein" and self.variant_type == "synonymous":
                mut_seq = wt_seq  # Same sequence
            elif self.modality == "dna":
                # DNA FASTA is centered on variant — inject at center
                center = len(wt_seq) // 2
                mut_seq = inject_variant_at_position(wt_seq, center, ref, alt)
            elif self.modality == "rna" and dna_sequences and var_id in dna_sequences:
                # RNA is gene-level mRNA — use DNA context to locate variant
                dna_seq = dna_sequences[var_id]
                result = find_variant_in_rna(wt_seq, dna_seq, ref)
                if result is not None:
                    rna_pos, is_minus = result
                    # Complement alt allele if gene is on minus strand
                    rna_alt = alt.translate(_COMPLEMENT) if is_minus else alt
                    mut_seq = inject_variant_at_position(wt_seq, rna_pos, ref, rna_alt)
                else:
                    # Could not locate variant in mRNA (intronic or edge case)
                    mut_seq = wt_seq
                    skipped_injection += 1
            else:
                # Fallback: center-based injection
                center = len(wt_seq) // 2
                mut_seq = inject_variant_at_position(wt_seq, center, ref, alt)

            self._wt_seqs.append(wt_seq)
            self._mut_seqs.append(mut_seq)
            self._labels.append(int(row["label"]))
            self._variant_ids.append(var_id)

        # Store metadata
        self._metadata = {
            "gene": df["gene"].tolist()[:len(self._variant_ids)],
            "ref": df["ref"].tolist()[:len(self._variant_ids)],
            "alt": df["alt"].tolist()[:len(self._variant_ids)],
        }

        # Log filtering stats
        if skipped_length > 0:
            print(f"Skipped {skipped_length} sequences exceeding max_seq_length={self.max_seq_length}")
        if skipped_injection > 0:
            print(f"WARNING: {skipped_injection} variants could not be injected into "
                  f"{self.modality} (variant not found in transcript)")

        n_diff = sum(1 for w, m in zip(self._wt_seqs, self._mut_seqs) if w != m)
        print(f"Variants with {self.modality} sequence change: "
              f"{n_diff}/{len(self._wt_seqs)} ({100*n_diff/len(self._wt_seqs):.1f}%)"
              if self._wt_seqs else "")

        # Required by experiment.py: datamodule.train_dataset.data.shape
        dataset = VariantPairDataset(self._wt_seqs, self._mut_seqs, self._labels)
        self.train_dataset = dataset
        self.test_dataset = dataset

    def get_sequence_pairs(self) -> Dict[str, List[str]]:
        """Return paired WT/MUT sequences for encoding.

        Returns:
            Dict with 'wt' and 'mut' keys mapping to lists of sequences.
        """
        return {"wt": self._wt_seqs, "mut": self._mut_seqs}

    def get_sequences(self) -> Dict[str, List[str]]:
        """Return sequences as channels for BatchEncoder compatibility."""
        return self.get_sequence_pairs()

    def get_labels(self) -> np.ndarray:
        """Return pathogenicity labels (0=benign, 1=pathogenic)."""
        return np.array(self._labels, dtype=np.int64)

    def get_variant_ids(self) -> List[str]:
        """Return list of variant IDs in order."""
        return self._variant_ids

    def get_metadata(self) -> Dict[str, List]:
        """Return metadata dict aligned with variant order."""
        return self._metadata

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return self.train_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.train_dataloader()

    @property
    def num_variants(self) -> int:
        """Number of variants loaded."""
        return len(self._variant_ids)

    def __repr__(self) -> str:
        if not self._variant_ids:
            return f"VariantDataModule(variant_type={self.variant_type}, modality={self.modality}, not_setup=True)"
        return (
            f"VariantDataModule("
            f"variant_type={self.variant_type}, "
            f"modality={self.modality}, "
            f"n_variants={self.num_variants})"
        )
