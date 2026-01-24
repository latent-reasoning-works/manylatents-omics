"""
manylatents-dogma: Central dogma foundation model encoders for manylatents.

This package provides pretrained foundation model encoders for the three
levels of the central dogma:

    - DNA: Evo2 (StripedHyena2 architecture, 1B/7B/40B params)
    - RNA: Orthrus (Mamba-based, 4-track/6-track variants)
    - Protein: ESM3 (1.4B params)

These encoders transform biological sequences into dense embeddings that
can be used for downstream analysis with manylatents dimensionality
reduction algorithms.

Example:
    >>> from manylatents.dogma.encoders import ESM3Encoder, OrthrusEncoder, Evo2Encoder
    >>>
    >>> # Encode the same biological information across modalities
    >>> protein_emb = ESM3Encoder().encode("MKFGVRA")
    >>> rna_emb = OrthrusEncoder().encode("AUGAAGUUUGGCGUCCGUGCCUGA")
    >>> dna_emb = Evo2Encoder().encode("ATGAAGTTTGGCGTCCGTGCCTGA")

    >>> # Fusion: concatenate all three modalities
    >>> from manylatents.dogma.algorithms import CentralDogmaFusion
    >>> from manylatents.dogma.data import CentralDogmaDataModule
"""

from . import encoders
from . import algorithms
from . import data

__version__ = "0.1.0"

__all__ = ["encoders", "algorithms", "data"]
