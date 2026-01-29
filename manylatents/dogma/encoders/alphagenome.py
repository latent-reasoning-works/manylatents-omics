"""AlphaGenome encoder for DNA sequences with regulatory predictions.

AlphaGenome is a JAX-based foundation model for genomics that predicts
regulatory features at single base-pair resolution across 1Mb context.

References:
    - Paper: "AlphaGenome: Foundation model for the human genome"
    - GitHub: https://github.com/google-deepmind/alphagenome_research
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from manylatents.algorithms.encoder import FoundationEncoder


class AlphaGenomeEncoder(FoundationEncoder):
    """AlphaGenome encoder for DNA sequences.

    Provides both embeddings (encode) and regulatory track predictions (predict).
    Uses JAX internally with transparent PyTorch tensor conversion.

    Args:
        model_name: Model variant. Options: "alphagenome" (1bp), "alphagenome_128bp".
        layer_name: Embedding resolution. Options: "embeddings_1bp", "embeddings_128bp".
        weights_path: Local path to weights. If None, downloads from HuggingFace.
        device: Device for PyTorch output tensors ("cuda" or "cpu").

    Example:
        >>> encoder = AlphaGenomeEncoder()
        >>> embedding = encoder.encode("ATGAAGTTTGGCGTCCGTGCCTGA")
        >>> predictions = encoder.predict("ATGAAGTTTGGCGTCCGTGCCTGA")
    """

    # Model configurations from architecture triage
    MODELS = {
        "alphagenome": {
            "default_layer": "embeddings_1bp",
            "embedding_dim": 1536,
            "context_length": 1_000_000,
        },
        "alphagenome_128bp": {
            "default_layer": "embeddings_128bp",
            "embedding_dim": 3072,
            "context_length": 1_000_000,
        },
    }

    DEFAULT_WEIGHTS = "/network/weights/alphagenome/"

    def __init__(
        self,
        model_name: str = "alphagenome",
        layer_name: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        if model_name not in self.MODELS:
            raise ValueError(
                f"model_name must be one of {list(self.MODELS.keys())}, got {model_name}"
            )

        self.model_name = model_name
        self.layer_name = layer_name or self.MODELS[model_name]["default_layer"]
        self.weights_path = weights_path
        self._embedding_dim = self.MODELS[model_name]["embedding_dim"]
        self._context_length = self.MODELS[model_name]["context_length"]

        self._model = None
        self._jax_to_torch = None
        self._torch_to_jax = None

    def _load_model(self) -> None:
        """Lazy load AlphaGenome JAX model."""
        if self._model is not None:
            return

        try:
            from alphagenome_research.model import dna_model
            from torch_jax_interop import jax_to_torch, torch_to_jax
        except ImportError as e:
            raise ImportError(
                "AlphaGenome requires 'alphagenome-research' and 'torch-jax-interop'. "
                "Install with: uv sync --extra alphagenome"
            ) from e

        self._jax_to_torch = jax_to_torch
        self._torch_to_jax = torch_to_jax
        self._dna_model = dna_model

        # Try local weights first, fall back to HuggingFace
        if self.weights_path and Path(self.weights_path).exists():
            self._model = dna_model.create_from_huggingface(self.weights_path)
        else:
            self._model = dna_model.create_from_huggingface("all_folds")

    def encode(
        self,
        sequence: str,
        chunk_size: Optional[int] = None,
        overlap: int = 0,
        aggregate: Optional[str] = "mean",
    ) -> Tensor:
        """Encode DNA sequence into embedding space.

        Args:
            sequence: DNA nucleotide sequence (ACGT alphabet).
            chunk_size: Process in chunks of this size. If None, process whole sequence.
            overlap: Overlap between chunks in base pairs.
            aggregate: How to combine chunks ("mean", "max", or None for raw chunks).

        Returns:
            Embedding tensor of shape (1, embedding_dim) if aggregated,
            or (num_chunks, embedding_dim) if aggregate=None.
        """
        self._ensure_loaded()

        # Short sequence - no chunking needed
        if not chunk_size or len(sequence) <= chunk_size:
            jax_emb = self._forward_embedding(sequence)
            return self._jax_to_torch(jax_emb).to(self.device)

        # Chunked encoding
        chunk_embeddings = self._encode_chunked(sequence, chunk_size, overlap)

        if aggregate is None:
            return chunk_embeddings
        elif aggregate == "mean":
            return chunk_embeddings.mean(dim=0, keepdim=True)
        elif aggregate == "max":
            return chunk_embeddings.max(dim=0, keepdim=True).values
        else:
            raise ValueError(f"aggregate must be 'mean', 'max', or None, got {aggregate}")

    def predict(
        self,
        sequence: str,
        chunk_size: Optional[int] = None,
        overlap: int = 0,
        aggregate: Optional[str] = None,
        output_types: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """Predict regulatory tracks for DNA sequence.

        Args:
            sequence: DNA nucleotide sequence (ACGT alphabet).
            chunk_size: Process in chunks of this size. If None, process whole sequence.
            overlap: Overlap between chunks in base pairs.
            aggregate: How to combine chunks (None recommended for position-specific tracks).
            output_types: List of output types to request. If None, returns all available.
                Options: "ATAC", "CAGE", "CHIP_HISTONE", "CHIP_TF", "DNASE", "RNA_SEQ", etc.

        Returns:
            Dict mapping track names to PyTorch tensors.
        """
        self._ensure_loaded()

        # Short sequence - no chunking needed
        if not chunk_size or len(sequence) <= chunk_size:
            jax_preds = self._forward_predict(sequence, output_types)
            return {
                name: self._jax_to_torch(arr).to(self.device)
                for name, arr in jax_preds.items()
            }

        # Chunked prediction
        return self._predict_chunked(sequence, chunk_size, overlap, aggregate, output_types)

    def _forward_embedding(self, sequence: str) -> Any:
        """Run forward pass and extract embedding from specified layer.

        Args:
            sequence: DNA sequence.

        Returns:
            JAX array of embeddings, mean-pooled over sequence length.
            Shape: (1, embedding_dim)
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        # Access the internal apply_fn to get raw predictions including embeddings
        # The public API filters out embeddings, so we need to call apply_fn directly
        apply_fn = self._model._predict.keywords.get("apply_fn")
        if apply_fn is None:
            raise RuntimeError(
                "Could not access internal apply_fn. AlphaGenome API may have changed."
            )

        # Encode sequence to one-hot
        one_hot = self._model._one_hot_encoder.encode(sequence)
        one_hot = jnp.asarray(one_hot)[jnp.newaxis]  # Add batch dim: (1, S, 4)

        # Create organism index (default to human = 0)
        organism_index = jnp.array([0], dtype=jnp.int32)

        # Call apply_fn to get raw predictions including embeddings_1bp
        predictions = apply_fn(
            self._model._params,
            self._model._state,
            one_hot,
            organism_index,
        )

        # Extract embeddings based on layer_name
        if self.layer_name == "embeddings_1bp":
            embeddings = predictions.get("embeddings_1bp")
            if embeddings is None:
                raise KeyError(
                    f"embeddings_1bp not found in predictions. "
                    f"Available keys: {list(predictions.keys())}"
                )
            # Mean pool over sequence length: (1, S, 1536) -> (1, 1536)
            return jnp.mean(embeddings, axis=1)

        elif self.layer_name == "embeddings_128bp":
            # For 128bp resolution, we'd need to access trunk embeddings
            # This requires different extraction logic
            raise NotImplementedError(
                "embeddings_128bp extraction not yet implemented. "
                "Use layer_name='embeddings_1bp' for now."
            )

        else:
            raise ValueError(
                f"Unknown layer_name: {self.layer_name}. "
                f"Supported: 'embeddings_1bp', 'embeddings_128bp'"
            )

    def _forward_predict(
        self, sequence: str, output_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run forward pass and return all prediction tracks.

        Args:
            sequence: DNA sequence.
            output_types: Optional list of output type names to request.
                Options: "ATAC", "CAGE", "CHIP_HISTONE", "CHIP_TF",
                         "DNASE", "RNA_SEQ", "SPLICE_SITES", etc.

        Returns:
            Dict of track name (lowercase) to JAX array.
        """
        OutputType = self._dna_model.OutputType

        if output_types is None:
            # Default to commonly used output types
            requested = [
                OutputType.ATAC,
                OutputType.CAGE,
                OutputType.DNASE,
                OutputType.RNA_SEQ,
            ]
        else:
            requested = [OutputType[ot.upper()] for ot in output_types]

        output = self._model.predict_sequence(
            sequence,
            requested_outputs=requested,
            ontology_terms=None,
        )

        results = {}
        for ot in requested:
            track_data = output.get(ot)
            if track_data is not None and track_data.data is not None:
                results[ot.name.lower()] = track_data.data

        return results

    def _encode_chunked(
        self,
        sequence: str,
        chunk_size: int,
        overlap: int,
    ) -> Tensor:
        """Encode long sequence in overlapping chunks."""
        chunks = self._split_sequence(sequence, chunk_size, overlap)
        embeddings = [self._forward_embedding(chunk) for chunk in chunks]

        import jax.numpy as jnp

        stacked = jnp.stack(embeddings, axis=0)
        return self._jax_to_torch(stacked).to(self.device)

    def _predict_chunked(
        self,
        sequence: str,
        chunk_size: int,
        overlap: int,
        aggregate: Optional[str],
        output_types: Optional[List[str]],
    ) -> Dict[str, Tensor]:
        """Predict on long sequence in overlapping chunks."""
        chunks = self._split_sequence(sequence, chunk_size, overlap)
        chunk_preds = [self._forward_predict(chunk, output_types) for chunk in chunks]

        if not chunk_preds:
            return {}

        track_names = chunk_preds[0].keys()
        import jax.numpy as jnp

        result = {}
        for name in track_names:
            stacked = jnp.stack([p[name] for p in chunk_preds], axis=0)
            tensor = self._jax_to_torch(stacked).to(self.device)

            if aggregate == "mean":
                tensor = tensor.mean(dim=0, keepdim=True)
            elif aggregate == "max":
                tensor = tensor.max(dim=0, keepdim=True).values
            # else: keep as (num_chunks, ...)

            result[name] = tensor

        return result

    @staticmethod
    def _split_sequence(sequence: str, chunk_size: int, overlap: int) -> List[str]:
        """Split sequence into overlapping chunks."""
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

        step = chunk_size - overlap
        chunks = []
        for i in range(0, len(sequence), step):
            chunk = sequence[i : i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)
        return chunks

    @property
    def modality(self) -> str:
        return "dna"

    @property
    def context_length(self) -> int:
        """Maximum sequence length the model can process."""
        return self._context_length
