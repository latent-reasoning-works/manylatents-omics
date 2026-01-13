"""Evo2 encoder for DNA sequences.

Evo 2 is a state-of-the-art DNA language model using the StripedHyena 2 architecture.
It models DNA sequences at single-nucleotide resolution with up to 1M base pair
context length. Available in 1B, 7B, and 40B parameter variants.

References:
    - Paper: Nguyen et al. (2025) "Genome modeling and design across all domains of life with Evo 2"
    - GitHub: https://github.com/ArcInstitute/evo2
    - PyPI: https://pypi.org/project/evo2/
"""

from typing import Any, List, Optional

import torch
from torch import Tensor

from manylatents.algorithms.encoder import FoundationEncoder


class Evo2Encoder(FoundationEncoder):
    """Evo2 encoder for DNA sequences.

    Encodes DNA sequences into dense embeddings using the pretrained Evo2
    model. Extracts embeddings from an intermediate layer (recommended over
    final layer for downstream tasks).

    Args:
        model_name: Model variant. One of "evo2_1b_base", "evo2_7b", "evo2_40b".
        layer_name: Layer to extract embeddings from. If None, uses middle layer.
        device: Device for inference ("cuda" or "cpu").

    Example:
        >>> encoder = Evo2Encoder(model_name="evo2_1b_base")
        >>> embedding = encoder.encode("ATGAAGTTTGGCGTCCGTGCCTGA")
        >>> print(embedding.shape)  # torch.Size([1, 2048])
    """

    # Model configurations
    MODELS = {
        "evo2_1b_base": {
            "default_layer": "blocks.14.mlp.l3",  # Middle layer of 28
            "embedding_dim": 2048,
        },
        "evo2_7b": {
            "default_layer": "blocks.16.mlp.l3",  # Middle layer
            "embedding_dim": 4096,
        },
        "evo2_7b_base": {
            "default_layer": "blocks.16.mlp.l3",
            "embedding_dim": 4096,
        },
        "evo2_40b": {
            "default_layer": "blocks.32.mlp.l3",  # Middle layer
            "embedding_dim": 8192,
        },
        "evo2_40b_base": {
            "default_layer": "blocks.32.mlp.l3",
            "embedding_dim": 8192,
        },
    }

    # Default weights location on Mila cluster
    DEFAULT_WEIGHTS = "/network/weights/savanna-evo2-1b-base/savanna_evo2_1b_base/savanna_evo2_1b_base.pt"

    def __init__(
        self,
        model_name: str = "evo2_1b_base",
        layer_name: Optional[str] = None,
        device: str = "cuda",
    ):
        super().__init__(device=device)

        if model_name not in self.MODELS:
            raise ValueError(
                f"model_name must be one of {list(self.MODELS.keys())}, got {model_name}"
            )

        self.model_name = model_name
        self.layer_name = layer_name or self.MODELS[model_name]["default_layer"]
        self._embedding_dim = self.MODELS[model_name]["embedding_dim"]

        self._model = None

    def _load_model(self):
        """Lazy load the Evo2 model."""
        if self._model is not None:
            return

        try:
            from evo2 import Evo2

            self._model = Evo2(self.model_name)

        except ImportError as e:
            raise ImportError(
                "Evo2 requires the 'evo2' package. Install with: pip install evo2"
            ) from e

    def encode(self, sequence: str) -> Tensor:
        """Encode a DNA sequence into embedding space.

        Args:
            sequence: DNA nucleotide sequence (e.g., "ATGAAGTTTGGCGTCCGTGCCTGA").
                      Should use ACGT alphabet.

        Returns:
            Embedding tensor of shape (1, embedding_dim).
        """
        self._load_model()

        # Tokenize the sequence
        input_ids = torch.tensor(
            self._model.tokenizer.tokenize(sequence),
            dtype=torch.int,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get embeddings from specified layer
            _, embeddings = self._model(
                input_ids,
                return_embeddings=True,
                layer_names=[self.layer_name],
            )

            # Mean pool over sequence length
            # embeddings[layer_name] shape: (1, seq_len, hidden_dim)
            embedding = embeddings[self.layer_name].mean(dim=1)

        return embedding

    def encode_batch(self, sequences: List[str]) -> Tensor:
        """Encode multiple DNA sequences.

        Note: Currently processes sequences one at a time due to variable lengths.
        For large batches, consider padding and batching manually.

        Args:
            sequences: List of DNA nucleotide sequences.

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        self._load_model()

        embeddings = []
        for seq in sequences:
            emb = self.encode(seq)
            embeddings.append(emb.squeeze(0))

        return torch.stack(embeddings, dim=0)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def modality(self) -> str:
        return "dna"
