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
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS
        self._model = None
        self._embedding_dim = 1536  # ESM3-small hidden dim

    def _load_model(self):
        """Lazy load the ESM3 model."""
        if self._model is not None:
            return

        try:
            from esm.models.esm3 import ESM3
            import os

            # Use local weights if path exists
            if self.weights_path and os.path.exists(self.weights_path):
                # Set HF cache to local weights directory for ESM3's data_root()
                weights_root = os.path.dirname(os.path.dirname(os.path.dirname(self.weights_path)))
                os.environ["HF_HOME"] = weights_root
                os.environ["HF_HUB_OFFLINE"] = "1"
                self._model = ESM3.from_pretrained("esm3_sm_open_v1")
            else:
                # Load from HuggingFace (requires login for gated model)
                self._model = ESM3.from_pretrained("esm3_sm_open_v1")

            # Keep model in float32 to avoid dtype mismatch bug in ESM3 library
            self._model = self._model.to(self.device).float().eval()

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
        self._ensure_loaded()

        from esm.sdk.api import ESMProtein

        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            # Tokenize and run forward pass for embeddings
            protein_tensor = self._model.encode(protein)
            # Add batch dimension for forward pass
            seq_tokens = protein_tensor.sequence.unsqueeze(0).to(self._model.device)
            output = self._model.forward(sequence_tokens=seq_tokens)
            # output.embeddings shape: (1, seq_len, hidden_dim)
            embedding = output.embeddings.mean(dim=1).float()

        return embedding

    # encode_batch() uses base class default (loops over encode())

    @property
    def modality(self) -> str:
        return "protein"
