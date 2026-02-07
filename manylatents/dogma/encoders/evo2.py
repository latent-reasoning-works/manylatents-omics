"""Evo2 encoder for DNA sequences.

Evo 2 is a state-of-the-art DNA language model using the StripedHyena 2 architecture.
It models DNA sequences at single-nucleotide resolution with up to 1M base pair
context length. Available in 1B, 7B, and 40B parameter variants.

References:
    - Paper: Nguyen et al. (2025) "Genome modeling and design across all domains of life with Evo 2"
    - GitHub: https://github.com/ArcInstitute/evo2
    - PyPI: https://pypi.org/project/evo2/
"""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from manylatents.dogma.encoders.base import FoundationEncoder


class Evo2Encoder(FoundationEncoder):
    """Evo2 encoder for DNA sequences.

    Encodes DNA sequences into dense embeddings using the pretrained Evo2
    model. Supports extracting from single or multiple layers simultaneously.

    When multiple layers are requested (multi_layer=True), encode() returns
    a dict mapping layer names to tensors. When a single layer is used
    (backward compat), encode() returns a flat tensor.

    Args:
        model_name: Model variant. One of "evo2_1b_base", "evo2_7b", "evo2_40b".
        layer_name: Single layer to extract from (backward compat). Overridden by layer_names.
        layer_names: List of layers to extract from. Returns dict output.
        device: Device for inference ("cuda" or "cpu").

    Example:
        >>> encoder = Evo2Encoder(model_name="evo2_1b_base")
        >>> result = encoder.encode("ATGAAGTTTGGCGTCCGTGCCTGA")
        >>> # Multi-layer default: result is dict with 3 layer keys
    """

    # Model configurations
    MODELS = {
        "evo2_1b_base": {
            "default_layer": "blocks.14.mlp.l3",  # Middle layer of 25
            "default_layers": [
                "blocks.14.mlp.l3",   # Middle (56%) — local features
                "blocks.19.mlp.l3",   # Late (76%) — functional features (Goodfire ~75%)
                "blocks.23.mlp.l3",   # Near-final (92%) — abstract representations
            ],
            "embedding_dim": 1920,  # hidden_size from model config
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
        layer_names: Optional[List[str]] = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        if model_name not in self.MODELS:
            raise ValueError(
                f"model_name must be one of {list(self.MODELS.keys())}, got {model_name}"
            )

        self.model_name = model_name
        self._embedding_dim = self.MODELS[model_name]["embedding_dim"]
        self._model = None

        # Layer selection: layer_names (list) > layer_name (str) > default
        if layer_names:
            self._layer_names = layer_names
            self._multi_layer = True
        elif layer_name:
            self._layer_names = [layer_name]
            self._multi_layer = False
        elif "default_layers" in self.MODELS[model_name]:
            self._layer_names = self.MODELS[model_name]["default_layers"]
            self._multi_layer = True
        else:
            self._layer_names = [self.MODELS[model_name]["default_layer"]]
            self._multi_layer = False

    @property
    def layer_names(self) -> List[str]:
        return self._layer_names

    @property
    def multi_layer(self) -> bool:
        return self._multi_layer

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

    def _pool_embeddings(
        self,
        embeddings: Dict[str, Tensor],
        mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Mean-pool layer embeddings, optionally with attention mask.

        Args:
            embeddings: Raw hidden states keyed by layer name.
            mask: Optional (B, L) attention mask for padded sequences.

        Returns:
            Dict of pooled tensors if multi_layer, else single pooled tensor.
        """
        def _pool_one(hidden: Tensor) -> Tensor:
            if mask is not None:
                m = mask.unsqueeze(-1).float().to(hidden.device)
                return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            return hidden.mean(dim=1)

        if self._multi_layer:
            return {name: _pool_one(embeddings[name]) for name in self._layer_names}
        return _pool_one(embeddings[self._layer_names[0]])

    def encode(self, sequence: str) -> Union[Tensor, Dict[str, Tensor]]:
        """Encode a DNA sequence into embedding space.

        Args:
            sequence: DNA nucleotide sequence (e.g., "ATGAAGTTTGGCGTCCGTGCCTGA").

        Returns:
            If multi_layer: dict mapping layer names to (1, embedding_dim) tensors.
            If single layer: (1, embedding_dim) tensor.
        """
        self._ensure_loaded()

        input_ids = torch.tensor(
            self._model.tokenizer.tokenize(sequence),
            dtype=torch.int,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, embeddings = self._model(
                input_ids,
                return_embeddings=True,
                layer_names=self._layer_names,
            )
            return self._pool_embeddings(embeddings)

    # --- Batched inference ---

    def _supports_batched_forward(self) -> bool:
        return True

    def _tokenize_batch(self, sequences: List[str]) -> dict:
        """Tokenize, pad, and stack sequences for batched forward pass."""
        self._ensure_loaded()

        encoded = [self._model.tokenizer.tokenize(seq) for seq in sequences]
        max_len = max(len(e) for e in encoded)

        input_ids = torch.zeros(len(encoded), max_len, dtype=torch.int,
                                device=self.device)
        attention_mask = torch.zeros(len(encoded), max_len, dtype=torch.bool,
                                     device=self.device)

        for i, enc in enumerate(encoded):
            length = len(enc)
            input_ids[i, :length] = torch.tensor(enc, dtype=torch.int)
            attention_mask[i, :length] = True

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _extract_embeddings(self, batch: dict) -> Union[Tensor, Dict[str, Tensor]]:
        """Single forward pass with masked mean pooling."""
        _, embeddings = self._model(
            batch["input_ids"],
            return_embeddings=True,
            layer_names=self._layer_names,
        )
        return self._pool_embeddings(embeddings, mask=batch["attention_mask"])

    @property
    def modality(self) -> str:
        return "dna"
