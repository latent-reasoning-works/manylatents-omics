"""Pre-computed variant effect scorers.

Scorers return per-variant pathogenicity predictions from external databases.
Unlike encoders, they produce scalars (not embeddings) and require no GPU.
"""

from manylatents.dogma.scorers.alphamissense import AlphaMissenseScorer

__all__ = ["AlphaMissenseScorer"]
