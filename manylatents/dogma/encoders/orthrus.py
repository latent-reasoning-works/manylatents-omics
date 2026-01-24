"""Orthrus encoder for RNA sequences.

Orthrus is a Mamba-based mature RNA foundation model pretrained via contrastive
learning on 45M+ transcripts from splice isoforms and orthologous genes across
400+ mammalian species.

Two variants are available:
    - 4-track (base): One-hot encoding only (256-dim embeddings)
    - 6-track (large): Adds splice site + CDS markers (512-dim embeddings)

References:
    - Paper: Fradkin et al. (2024) "Orthrus: Towards Evolutionary and Functional RNA Foundation Models"
    - HuggingFace: https://huggingface.co/antichronology/orthrus
    - Zenodo: https://zenodo.org/records/13910050
"""

from typing import Any, List, Optional

import torch
from torch import Tensor

from manylatents.algorithms.encoder import FoundationEncoder


class OrthrusEncoder(FoundationEncoder):
    """Orthrus encoder for mature RNA sequences.

    Encodes RNA sequences into dense embeddings using the pretrained Orthrus
    model. Supports both 4-track (simple) and 6-track (with splice/CDS info)
    variants.

    Args:
        n_tracks: Number of input tracks (4 or 6). Default 4.
        run_path: Path to model directory. If None, uses default.
        checkpoint: Checkpoint filename. If None, uses default for n_tracks.
        device: Device for inference ("cuda" or "cpu").

    Example:
        >>> encoder = OrthrusEncoder(n_tracks=4)
        >>> embedding = encoder.encode("AUGAAGUUUGGCGUCCGUGCCUGA")
        >>> print(embedding.shape)  # torch.Size([1, 256])
    """

    # Default model paths on Mila cluster
    MODELS = {
        4: {
            "run_path": "/network/weights/orthrus/Orthrus/models/orthrus_base_4_track",
            "checkpoint": "epoch=18-step=20000.ckpt",
            "embedding_dim": 256,
        },
        6: {
            "run_path": "/network/weights/orthrus/Orthrus/models/orthrus_large_6_track",
            "checkpoint": "epoch=22-step=20000.ckpt",
            "embedding_dim": 512,
        },
    }

    def __init__(
        self,
        n_tracks: int = 4,
        run_path: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        if n_tracks not in self.MODELS:
            raise ValueError(f"n_tracks must be 4 or 6, got {n_tracks}")

        self.n_tracks = n_tracks
        self.run_path = run_path or self.MODELS[n_tracks]["run_path"]
        self.checkpoint = checkpoint or self.MODELS[n_tracks]["checkpoint"]
        self._embedding_dim = self.MODELS[n_tracks]["embedding_dim"]

        self._model = None
        self._seq_to_oh = None

    def _load_model(self):
        """Lazy load the Orthrus model."""
        if self._model is not None:
            return

        import sys

        # Compatibility shim: mamba-ssm 2.x moved Block to mamba_ssm.modules.block
        # Orthrus expects it in mamba_ssm.modules.mamba_simple
        try:
            from mamba_ssm.modules.mamba_simple import Block
        except ImportError:
            from mamba_ssm.modules.block import Block
            import mamba_ssm.modules.mamba_simple as mamba_simple
            mamba_simple.Block = Block

        # Add Orthrus to path for imports
        orthrus_path = "/network/weights/orthrus/Orthrus"
        if orthrus_path not in sys.path:
            sys.path.insert(0, orthrus_path)

        try:
            from orthrus.eval_utils import load_model
            from orthrus.sequence import seq_to_oh

            self._model = load_model(self.run_path, self.checkpoint)
            self._model = self._model.to(self.device).eval()
            self._seq_to_oh = seq_to_oh

        except ImportError as e:
            raise ImportError(
                f"Failed to import Orthrus from {orthrus_path}. "
                "Ensure the Orthrus repository is available."
            ) from e

    def encode(self, sequence: str) -> Tensor:
        """Encode an RNA sequence into embedding space.

        Args:
            sequence: RNA nucleotide sequence (e.g., "AUGAAGUUUGGCGUCCGUGCCUGA").
                      Should use U (uracil), not T (thymine).

        Returns:
            Embedding tensor of shape (1, embedding_dim).
        """
        self._ensure_loaded()

        # Convert sequence to one-hot encoding
        # seq_to_oh returns shape (n_tracks, seq_len), we need (batch, seq_len, n_tracks)
        oh = self._seq_to_oh(sequence)
        oh = torch.tensor(oh.T, dtype=torch.float32)  # (seq_len, n_tracks)
        oh = oh.unsqueeze(0).to(self.device)  # (1, seq_len, n_tracks)

        lengths = torch.tensor([oh.shape[1]], device=self.device)

        with torch.no_grad():
            embedding = self._model.representation(oh, lengths, channel_last=True)

        return embedding

    # encode_batch() uses base class default (loops over encode())

    @property
    def modality(self) -> str:
        return "rna"
