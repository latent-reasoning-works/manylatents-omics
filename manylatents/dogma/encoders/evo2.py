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

from manylatents.algorithms.latent.foundation_encoder import FoundationEncoder


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
        weights_path: Explicit checkpoint path, passed to Evo2 as local_path.
            Offline clusters cannot resolve by name. Falls back to
            EVO2_WEIGHTS_DIR/<model_name> when that directory exists.
        device: Device for inference ("cuda" or "cpu").

    Pooling is masked mean over non-padding positions, accumulated in float32
    on both the single-sequence and batched paths.

    Example:
        >>> encoder = Evo2Encoder(model_name="evo2_1b_base")
        >>> result = encoder.encode("ATGAAGTTTGGCGTCCGTGCCTGA")
        >>> # Multi-layer default: result is dict with 3 layer keys
    """

    # Model configurations.
    #
    # Upstream block depths are 25 (1B), 32 (7B) and 50 (40B). Every size
    # declares the same three-layer profile so the return shape does not depend
    # on which model you picked; previously only the 1B had `default_layers`, so
    # the 1B returned a dict of three tensors and 7B/40B a single tensor, and a
    # tensor-based operator builder broke on the dict.
    #
    # No default sits at or near the final block. Evo2-7B shows build-and-flush:
    # informative geometry at intermediate layers, numerical annihilation at the
    # last one. The percentages below are of total depth.
    MODELS = {
        "evo2_1b_base": {
            "depth": 25,
            "default_layer": "blocks.14.mlp.l3",   # 56%
            "default_layers": [
                "blocks.14.mlp.l3",   # 56% — local features
                "blocks.19.mlp.l3",   # 76% — functional features (Goodfire ~75%)
                "blocks.23.mlp.l3",   # 92% — abstract representations
            ],
            "embedding_dim": 1920,
        },
        "evo2_7b": {
            "depth": 32,
            "default_layer": "blocks.16.mlp.l3",   # 50%
            "default_layers": [
                "blocks.16.mlp.l3",   # 50%
                "blocks.24.mlp.l3",   # 75%
                "blocks.29.mlp.l3",   # 91%
            ],
            "embedding_dim": 4096,
        },
        "evo2_7b_base": {
            "depth": 32,
            "default_layer": "blocks.16.mlp.l3",
            "default_layers": [
                "blocks.16.mlp.l3",
                "blocks.24.mlp.l3",
                "blocks.29.mlp.l3",
            ],
            "embedding_dim": 4096,
        },
        "evo2_40b": {
            "depth": 50,
            # Was blocks.32, labelled "middle layer". That is the 33rd of 50 — 66%.
            "default_layer": "blocks.25.mlp.l3",   # 50%
            "default_layers": [
                "blocks.25.mlp.l3",   # 50%
                "blocks.37.mlp.l3",   # 74%
                "blocks.46.mlp.l3",   # 92%
            ],
            "embedding_dim": 8192,
        },
        "evo2_40b_base": {
            "depth": 50,
            "default_layer": "blocks.25.mlp.l3",
            "default_layers": [
                "blocks.25.mlp.l3",
                "blocks.37.mlp.l3",
                "blocks.46.mlp.l3",
            ],
            "embedding_dim": 8192,
        },
    }

    def __init__(
        self,
        model_name: str = "evo2_1b_base",
        layer_name: Optional[str] = None,
        layer_names: Optional[List[str]] = None,
        weights_path: Optional[str] = None,
        trainable: bool = False,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        if model_name not in self.MODELS:
            raise ValueError(
                f"model_name must be one of {list(self.MODELS.keys())}, got {model_name}"
            )

        self.model_name = model_name
        # Offline clusters cannot resolve by name; EVO2_WEIGHTS_DIR lets a site
        # set one directory rather than threading a path through every config.
        self.weights_path = weights_path or self._weights_from_environment(model_name)
        self.trainable = trainable
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

    def _dematerialise_inference_tensors(self) -> tuple[int, int]:
        """Replace every parameter and buffer with an ordinary autograd tensor.

        vortex/model/utils.py:102 wraps torch.load and the weight copy in
        `with torch.inference_mode():`, so every loaded weight is an inference
        tensor permanently. Such a tensor cannot be saved for backward whatever
        its requires_grad flag says -- and Evo2-1B reports requires_grad=True on
        all 1.108B parameters, which is why the flag is no guide. Cloning
        outside that context is PyTorch's documented escape hatch.

        Buffers are not optional. compute_filter evaluates
        `(residues[..., None] * (log_poles * self.t).exp()).sum(1)` and `self.t`
        is a buffer, so one inference operand poisons the graph even with clean
        parameters. Spike 465441 converted 265 parameters and 4 buffers; see
        docs/evo2_trainability.md.

        Returns the counts, so a caller can assert the walk reached something.
        """
        inner = getattr(self._model, "model", self._model)
        parameters = buffers = 0
        for module in inner.modules():
            for name, param in list(module._parameters.items()):
                if param is None:
                    continue
                module._parameters[name] = torch.nn.Parameter(
                    param.detach().clone(), requires_grad=param.requires_grad
                )
                parameters += 1
            for name, buffer in list(module._buffers.items()):
                if buffer is None:
                    continue
                module._buffers[name] = buffer.detach().clone()
                buffers += 1
        return parameters, buffers

    def _forward_capturing(self, input_ids: Tensor) -> Dict[str, Tensor]:
        """Forward with hooks that do NOT detach, bypassing Evo2.forward.

        Evo2.forward wraps the model call in `with torch.no_grad():` and its own
        embedding hook stores `output.detach()`. Either alone severs the graph,
        so the trainable path calls the inner StripedHyena directly.
        """
        inner = getattr(self._model, "model", self._model)
        captured: Dict[str, Tensor] = {}
        modules = dict(inner.named_modules())
        handles = []

        def _make_hook(layer: str):
            def hook(_module, _inputs, output):
                captured[layer] = output[0] if isinstance(output, tuple) else output
            return hook

        try:
            for layer in self._layer_names:
                target = modules.get(layer)
                if target is None:
                    raise KeyError(f"layer {layer!r} does not resolve in {self.model_name}")
                handles.append(target.register_forward_hook(_make_hook(layer)))
            inner(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        missing = [name for name in self._layer_names if name not in captured]
        if missing:
            raise RuntimeError(f"no activations captured for {missing}")
        return captured

    @staticmethod
    def _weights_from_environment(model_name: str) -> Optional[str]:
        """The merged checkpoint under EVO2_WEIGHTS_DIR, or None.

        Evo2 passes `local_path` straight to load_checkpoint, which calls
        torch.load on it, so this must resolve to the *file* and never to the
        staging directory that contains it. The 40B stages as
        <root>/evo2_40b/{evo2_40b.pt, evo2_40b.pt.part0, evo2_40b.pt.part1},
        and returning that directory would hand torch.load a directory.
        """
        import os
        from pathlib import Path

        root = os.environ.get("EVO2_WEIGHTS_DIR")
        if not root:
            return None
        for candidate in (
            Path(root) / model_name / f"{model_name}.pt",
            Path(root) / f"{model_name}.pt",
        ):
            if candidate.is_file():
                return str(candidate)
        return None

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

            if self.weights_path:
                self._model = Evo2(self.model_name, local_path=self.weights_path)
            else:
                self._model = Evo2(self.model_name)

            # No active dropout was found, but a teacher cache is only reproducible
            # if repeated forwards agree, so say so rather than rely on it.
            inner = getattr(self._model, "model", self._model)
            if hasattr(inner, "eval"):
                inner.eval()

            if self.trainable:
                self._dematerialise_inference_tensors()

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
            # Accumulate and return float32 on both paths. encode() used to stay
            # bfloat16 while the batched path promoted via a .float() mask: two
            # points 0.0026 apart in fp32 collapse to exactly 0.0 in bf16, and the
            # operator is exp(-d^2 / 2 sigma^2) over cdist of these vectors, so
            # bf16 pooling erases the near-neighbour structure it is built from.
            hidden = hidden.float()
            if mask is not None:
                m = mask.unsqueeze(-1).to(dtype=hidden.dtype, device=hidden.device)
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

        if self.trainable:
            return self._pool_embeddings(self._forward_capturing(input_ids))

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

        # torch.zeros padded with 0, which is EOS/EOD. The pad id is 1.
        pad_id = getattr(self._model.tokenizer, "pad_id", 1)
        input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.int,
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
        if self.trainable:
            embeddings = self._forward_capturing(batch["input_ids"])
        else:
            _, embeddings = self._model(
                batch["input_ids"],
                return_embeddings=True,
                layer_names=self._layer_names,
            )
        return self._pool_embeddings(embeddings, mask=batch["attention_mask"])

    @property
    def modality(self) -> str:
        return "dna"
