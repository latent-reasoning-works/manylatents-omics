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

from manylatents.dogma.encoders.base import FoundationEncoder


class ESM3Encoder(FoundationEncoder):
    """ESM3 encoder for protein sequences.

    Encodes amino acid sequences into dense embeddings using the pretrained
    ESM3 model. Returns mean-pooled embeddings over the sequence length.

    Args:
        weights_path: Path to local weights. If None, loads from HuggingFace.
        device: Device for inference ("cuda" or "cpu").
        max_length: Maximum sequence length. Sequences longer than this will
            be truncated. If None, no truncation is applied.

    Example:
        >>> encoder = ESM3Encoder(max_length=2000)
        >>> embedding = encoder.encode("MKFGVRA")
        >>> print(embedding.shape)  # torch.Size([1, 1536])
    """

    # Default weights location on Mila cluster
    DEFAULT_WEIGHTS = "/network/weights/esm3-sm-open-v1/esm3-sm-open-v1/data/weights/esm3_sm_open_v1.pth"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS
        self.max_length = max_length
        self._model = None
        self._embedding_dim = 1536  # ESM3-small hidden dim

    def _load_model(self):
        """Lazy load the ESM3 model."""
        if self._model is not None:
            return

        try:
            from esm.models.esm3 import ESM3

            # Always load from HuggingFace - the ESM library handles caching
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

        if self.max_length is not None and len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]

        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            protein_tensor = self._model.encode(protein)
            seq_tokens = protein_tensor.sequence.unsqueeze(0).to(self._model.device)
            output = self._model.forward(sequence_tokens=seq_tokens)
            embedding = output.embeddings.mean(dim=1).float()

        return embedding

    # --- Batched inference ---

    def _supports_batched_forward(self) -> bool:
        return True

    def _tokenize_batch(self, sequences: List[str]) -> dict:
        """Tokenize, pad, and stack protein sequences for batched forward pass."""
        self._ensure_loaded()

        if self.max_length is not None:
            sequences = [seq[:self.max_length] for seq in sequences]

        tokenizer = self._model.tokenizers.sequence
        encoded = tokenizer(
            sequences,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded.input_ids.to(self.device),
            "attention_mask": encoded.attention_mask.to(self.device),
        }

    def _extract_embeddings(self, batch: dict) -> Tensor:
        """Single forward pass with masked mean pooling."""
        output = self._model.forward(sequence_tokens=batch["input_ids"])
        hidden = output.embeddings  # (B, L, 1536)

        mask = batch["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled  # (B, 1536)

    @property
    def modality(self) -> str:
        return "protein"
