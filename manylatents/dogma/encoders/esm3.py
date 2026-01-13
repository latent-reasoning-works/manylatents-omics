"""ESM3 encoder for protein sequences.

ESM3 is a frontier multimodal protein model from EvolutionaryScale that jointly
reasons across sequence, structure, and function. The open model (esm3-sm-open-v1)
has 1.4B parameters.

References:
    - Paper: Hayes et al. (2024) "Simulating 500 million years of evolution with a language model"
    - HuggingFace: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
"""

from typing import Any, List, Optional

import torch
from torch import Tensor

from manylatents.algorithms.encoder import FoundationEncoder


class ESM3Encoder(FoundationEncoder):
    """ESM3 encoder for protein sequences.

    Encodes amino acid sequences into dense embeddings using the pretrained
    ESM3 model. Returns mean-pooled embeddings over the sequence length.

    Args:
        weights_path: Path to local weights. If None, loads from HuggingFace.
        device: Device for inference ("cuda" or "cpu").

    Example:
        >>> encoder = ESM3Encoder()
        >>> embedding = encoder.encode("MKFGVRA")
        >>> print(embedding.shape)  # torch.Size([1, 1536])
    """

    # Default weights location on Mila cluster
    DEFAULT_WEIGHTS = "/network/weights/esm3-sm-open-v1/esm3-sm-open-v1/data/weights/esm3_sm_open_v1.pth"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
    ):
        super().__init__(device=device)
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS
        self._model = None
        self._embedding_dim = 1536  # ESM3-small hidden dim

    def _load_model(self):
        """Lazy load the ESM3 model."""
        if self._model is not None:
            return

        try:
            from esm.models.esm3 import ESM3

            # Try loading from local path first
            if self.weights_path and self.weights_path != self.DEFAULT_WEIGHTS:
                self._model = torch.load(self.weights_path, map_location=self.device)
            else:
                # Load from HuggingFace (requires login)
                self._model = ESM3.from_pretrained("esm3_sm_open_v1")

            self._model = self._model.to(self.device).eval()

        except ImportError as e:
            raise ImportError(
                "ESM3 requires the 'esm' package. Install with: pip install esm"
            ) from e

    def encode(self, sequence: str) -> Tensor:
        """Encode a protein sequence into embedding space.

        Args:
            sequence: Amino acid sequence (e.g., "MKFGVRA").

        Returns:
            Embedding tensor of shape (1, 1536).
        """
        self._load_model()

        from esm.sdk.api import ESMProtein

        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            # ESM3 returns per-residue embeddings, we mean-pool
            output = self._model.encode(protein)
            # output.embeddings shape: (1, seq_len, hidden_dim)
            embedding = output.embeddings.mean(dim=1)  # (1, hidden_dim)

        return embedding

    def encode_batch(self, sequences: List[str]) -> Tensor:
        """Encode multiple protein sequences.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            Embedding tensor of shape (batch_size, 1536).
        """
        self._load_model()

        from esm.sdk.api import ESMProtein

        embeddings = []
        with torch.no_grad():
            for seq in sequences:
                protein = ESMProtein(sequence=seq)
                output = self._model.encode(protein)
                emb = output.embeddings.mean(dim=1)
                embeddings.append(emb.squeeze(0))

        return torch.stack(embeddings, dim=0)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def modality(self) -> str:
        return "protein"
