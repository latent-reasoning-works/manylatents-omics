"""ESM encoder for protein sequences (ESM-1, ESM-1b, ESM-1v, ESM-2).

Unified wrapper for the Facebook Research ESM model family. All variants
share the same API: masked language model logits + per-layer representations.
ESM-2 650M is the default; ESM-1v ensemble is recommended for variant effect
prediction via masked marginals.

Requires the ``fair-esm`` package (NOT the EvolutionaryScale ``esm`` package
used by ESM3 — they share the ``esm`` namespace and cannot coexist).

    pip install fair-esm

References:
    - ESM-2: Lin et al. (2023) "Evolutionary-scale prediction of atomic-level
      protein structure with a language model"
    - ESM-1v: Meier et al. (2021) "Language models enable zero-shot prediction
      of the effects of mutations on protein function"
    - ESM-1b: Rives et al. (2021) "Biological structure and function emerge
      from scaling unsupervised learning to 250 million protein sequences"
    - Repo: https://github.com/facebookresearch/esm
"""

from typing import List, Optional

import torch
from torch import Tensor

from manylatents.algorithms.latent.foundation_encoder import FoundationEncoder


# Model registry: name → (num_layers, embedding_dim)
_ESM_MODELS = {
    # ESM-2 family (UniRef50/D, rotary embeddings, no hard positional limit)
    "esm2_t6_8M_UR50D": (6, 320),
    "esm2_t12_35M_UR50D": (12, 480),
    "esm2_t30_150M_UR50D": (30, 640),
    "esm2_t33_650M_UR50D": (33, 1280),
    "esm2_t36_3B_UR50D": (36, 2560),
    "esm2_t48_15B_UR50D": (48, 5120),
    # ESM-1v family (UniRef90/S, 5-model ensemble for variant prediction)
    "esm1v_t33_650M_UR90S_1": (33, 1280),
    "esm1v_t33_650M_UR90S_2": (33, 1280),
    "esm1v_t33_650M_UR90S_3": (33, 1280),
    "esm1v_t33_650M_UR90S_4": (33, 1280),
    "esm1v_t33_650M_UR90S_5": (33, 1280),
    # ESM-1b (UniRef50/S, the original widely-used model)
    "esm1b_t33_650M_UR50S": (33, 1280),
    # ESM-1 (earlier versions)
    "esm1_t34_670M_UR50S": (34, 1280),
    "esm1_t12_85M_UR50S": (12, 768),
    "esm1_t6_43M_UR50S": (6, 768),
}


class ESMEncoder(FoundationEncoder):
    """Unified encoder for the ESM protein language model family.

    Wraps ESM-1, ESM-1b, ESM-1v, and ESM-2 behind a single interface.
    Returns mean-pooled representations from a chosen layer, excluding
    special tokens (BOS/EOS/PAD).

    Args:
        model_name: Any model from the fair-esm registry. Default is the
            ESM-2 650M model (best general-purpose embeddings).
        repr_layer: Which transformer layer to extract from. None = final.
        device: Device for inference.
        max_length: Truncate sequences longer than this (default 1022).

    Example:
        >>> encoder = ESMEncoder()
        >>> embedding = encoder.encode("MKFGVRA")
        >>> print(embedding.shape)  # torch.Size([1, 1280])

        # ESM-1v for variant effect prediction
        >>> encoder = ESMEncoder(model_name="esm1v_t33_650M_UR90S_1")
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        repr_layer: Optional[int] = None,
        device: str = "cuda",
        max_length: int = 1022,
        **kwargs,
    ):
        if model_name not in _ESM_MODELS:
            raise ValueError(
                f"Unknown model {model_name!r}. "
                f"Available: {list(_ESM_MODELS.keys())}"
            )
        num_layers, embed_dim = _ESM_MODELS[model_name]
        super().__init__(device=device, **kwargs)
        self.model_name = model_name
        self.repr_layer = repr_layer if repr_layer is not None else num_layers
        self.max_length = max_length
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._embedding_dim = embed_dim

    def _load_model(self):
        """Lazy load model via fair-esm."""
        if self._model is not None:
            return

        try:
            import esm
            loader = getattr(esm.pretrained, self.model_name, None)
            if loader is None:
                raise AttributeError
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"ESMEncoder requires the 'fair-esm' package for {self.model_name}. "
                "Install with: pip install fair-esm\n"
                "Note: fair-esm conflicts with the ESM3 'esm' package."
            ) from e

        self._model, self._alphabet = loader()
        self._batch_converter = self._alphabet.get_batch_converter(
            truncation_seq_length=self.max_length,
        )
        self._model = self._model.to(self.device).eval()

    def encode(self, sequence: str) -> Tensor:
        """Encode a protein sequence into embedding space.

        Args:
            sequence: Amino acid sequence (e.g., "MKFGVRA").

        Returns:
            Embedding tensor of shape (1, embedding_dim).
        """
        self._ensure_loaded()

        _, _, batch_tokens = self._batch_converter([("_", sequence)])
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self._model(
                batch_tokens, repr_layers=[self.repr_layer]
            )

        token_repr = results["representations"][self.repr_layer]  # (1, L, D)
        # Mean-pool over residue positions, excluding BOS (0) and EOS/PAD
        seq_len = (batch_tokens != self._alphabet.padding_idx).sum(1)
        embedding = token_repr[0, 1 : seq_len[0] - 1].mean(0, keepdim=True)
        return embedding

    # --- Batched inference ---

    def _supports_batched_forward(self) -> bool:
        return True

    def _tokenize_batch(self, sequences: List[str]) -> dict:
        self._ensure_loaded()
        labels = [f"seq_{i}" for i in range(len(sequences))]
        data = list(zip(labels, sequences))
        _, _, batch_tokens = self._batch_converter(data)
        return {"tokens": batch_tokens.to(self.device)}

    def _extract_embeddings(self, batch: dict) -> Tensor:
        tokens = batch["tokens"]
        results = self._model(tokens, repr_layers=[self.repr_layer])
        token_repr = results["representations"][self.repr_layer]  # (B, L, D)

        # Per-sequence mean pool excluding special tokens
        mask = torch.ones_like(tokens, dtype=torch.float)
        mask[:, 0] = 0  # BOS
        lengths = (tokens != self._alphabet.padding_idx).sum(1)
        for i, l in enumerate(lengths):
            mask[i, l - 1] = 0  # EOS
            mask[i, l:] = 0  # PAD

        mask = mask.unsqueeze(-1)  # (B, L, 1)
        pooled = (token_repr * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled  # (B, D)

    @property
    def modality(self) -> str:
        return "protein"
