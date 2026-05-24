"""ESM3 encoder for protein sequences.

ESM3 is a frontier multimodal protein model from EvolutionaryScale that jointly
reasons across sequence, structure, and function. The open model (esm3-sm-open-v1)
has 1.4B parameters.

References:
    - Paper: Hayes et al. (2024) "Simulating 500 million years of evolution with a language model"
    - HuggingFace: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
"""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from manylatents.algorithms.latent.foundation_encoder import FoundationEncoder


class ESM3Encoder(FoundationEncoder):
    """ESM3 encoder for protein sequences.

    Encodes amino acid sequences into dense embeddings using the pretrained
    ESM3 model. Returns mean-pooled embeddings over the sequence length.

    Args:
        weights_path: Path to local weights. If None, loads from HuggingFace.
        device: Device for inference ("cuda" or "cpu").
        max_length: Maximum sequence length. Sequences longer than this will
            be truncated. If None, no truncation is applied.
        layer_indices: Absolute transformer-block indices to extract per-layer
            embeddings from (0..47 for esm3-sm-open-v1, which has 48 blocks).
            If given, encode() returns a dict {f"layer_{i}": (1, 1536)} keyed
            in this order; if None, encode() returns a single final-layer
            (1, 1536) tensor (backward compatible).
        reduce: Sequence-dim reduction for each layer. "mean" pools over all
            tokens (default, matches the prior behaviour and the LVD
            convention). "position" returns the embedding of a single residue
            and requires encode(sequence, position=...).

    Example:
        >>> encoder = ESM3Encoder(max_length=2000)
        >>> embedding = encoder.encode("MKFGVRA")
        >>> print(embedding.shape)  # torch.Size([1, 1536])
        >>> multi = ESM3Encoder(layer_indices=[0, 24, 47])
        >>> out = multi.encode("MKFGVRA")  # {"layer_0":..., "layer_24":..., "layer_47":...}
    """

    # Default weights location on Mila cluster
    DEFAULT_WEIGHTS = "/network/weights/esm3-sm-open-v1/esm3-sm-open-v1/data/weights/esm3_sm_open_v1.pth"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        max_length: Optional[int] = None,
        layer_indices: Optional[List[int]] = None,
        reduce: str = "mean",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS
        self.max_length = max_length
        self._model = None
        self._embedding_dim = 1536  # ESM3-small hidden dim

        # Per-layer extraction. ESM3.forward only exposes the final embedding
        # (TransformerStack builds a per-block `hiddens` list but ESM3 discards
        # it), so intermediate block outputs are captured via forward hooks on
        # transformer.blocks[i] — see _forward_hidden().
        self._layer_indices = list(layer_indices) if layer_indices else None
        self._multi_layer = bool(self._layer_indices)
        if reduce not in ("mean", "position"):
            raise ValueError(f"reduce must be 'mean' or 'position', got {reduce!r}")
        self._reduce = reduce

    @property
    def multi_layer(self) -> bool:
        return self._multi_layer

    @property
    def layer_indices(self) -> Optional[List[int]]:
        return self._layer_indices

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

    def _reduce_hidden(self, hidden: Tensor, position: Optional[int]) -> Tensor:
        """Reduce a (1, L, H) hidden state over the sequence dim → (1, H).

        ESM3 prepends a BOS token, so residue p (0-indexed) lives at token
        index p + 1 — matching the ESM-1b convention used elsewhere.
        """
        if self._reduce == "position":
            if position is None:
                raise ValueError(
                    "reduce='position' requires a residue index: call "
                    "encode(sequence, position=...) or use reduce='mean'."
                )
            return hidden[:, int(position) + 1, :].float()
        return hidden.mean(dim=1).float()

    def _forward_hidden(self, seq_tokens: Tensor) -> Dict[int, Tensor]:
        """Run a single ESM3 forward, capturing each requested transformer
        block's output via the shared ActivationExtractor. Returns
        {layer_idx: (B, L, H)}; pooling happens later in _reduce_hidden.

        ESM3.forward discards TransformerStack's per-block `hiddens` list, so
        hooks on transformer.blocks[i] are the access path (esm 3.2.1:
        UnifiedTransformerBlock.forward returns a plain tensor).
        """
        from manylatents.lightning.hooks import ActivationExtractor, LayerSpec

        paths = {i: f"transformer.blocks[{i}]" for i in self._layer_indices}
        specs = [LayerSpec(p, reduce="none") for p in paths.values()]
        acts = ActivationExtractor.extract_once(
            self._model,
            lambda: self._model.forward(sequence_tokens=seq_tokens),
            specs,
        )
        return {i: acts[p] for i, p in paths.items()}

    def encode(
        self, sequence: str, position: Optional[int] = None,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Encode a protein sequence into embedding space.

        Args:
            sequence: Amino acid sequence (e.g., "MKFGVRA").
            position: Residue index (0-based) for reduce='position'. Ignored
                when reduce='mean'.

        Returns:
            If layer_indices was set: dict {f"layer_{i}": (1, 1536)} in
            layer_indices order. Otherwise: a single (1, 1536) tensor from the
            final layer.
        """
        self._ensure_loaded()

        from esm.sdk.api import ESMProtein

        if self.max_length is not None and len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]

        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            protein_tensor = self._model.encode(protein)
            seq_tokens = protein_tensor.sequence.unsqueeze(0).to(self._model.device)

            if not self._multi_layer:
                output = self._model.forward(sequence_tokens=seq_tokens)
                return self._reduce_hidden(output.embeddings, position)

            captured = self._forward_hidden(seq_tokens)
            return {
                f"layer_{idx}": self._reduce_hidden(captured[idx], position)
                for idx in self._layer_indices
            }

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

    def _extract_embeddings(self, batch: dict) -> Union[Tensor, Dict[str, Tensor]]:
        """Single forward pass with masked mean pooling.

        Multi-layer (layer_indices set) captures each requested block via hooks
        and returns {f"layer_{i}": (B, 1536)}. reduce='position' is not
        supported in the batched path (per-sequence positions aren't threaded);
        use encode(sequence, position=...) one sequence at a time instead.
        """
        mask = batch["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)

        def _masked_mean(hidden: Tensor) -> Tensor:
            return ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).float()

        if not self._multi_layer:
            output = self._model.forward(sequence_tokens=batch["input_ids"])
            return _masked_mean(output.embeddings)  # (B, 1536)

        if self._reduce == "position":
            raise NotImplementedError(
                "reduce='position' is not supported with batched encoding; "
                "call encode(sequence, position=...) per sequence."
            )
        captured = self._forward_hidden(batch["input_ids"])
        return {f"layer_{idx}": _masked_mean(captured[idx]) for idx in self._layer_indices}

    @property
    def modality(self) -> str:
        return "protein"
