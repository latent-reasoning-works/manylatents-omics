"""Batch encoder for foundation model embedding generation.

Wraps a single foundation model encoder to process batches of sequences.
Designed for use with ClinVarDataModule which provides lists of sequences
per modality, unlike CentralDogmaDataModule which provides single sequences.

Usage:
    >>> encoder = BatchEncoder(
    ...     encoder_config={'_target_': 'manylatents.dogma.encoders.Evo2Encoder'},
    ...     modality='dna',
    ...     datamodule=clinvar_datamodule,
    ... )
    >>> embeddings = encoder.fit_transform(dummy_tensor)
    >>> print(embeddings.shape)  # (N, 1920) where N = number of variants

For ClinVar pipeline, this enables parallel encoding jobs:
    python -m manylatents.main +experiment=clinvar/encode_dna   # Evo2
    python -m manylatents.main +experiment=clinvar/encode_protein  # ESM3
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import hydra
import torch
from torch import Tensor

from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


class BatchEncoder(LatentModule):
    """Encode batches of sequences using a single foundation model.

    Takes sequences from datamodule.get_sequences()[modality] (a list of strings)
    and encodes each sequence, returning stacked embeddings.

    Args:
        encoder_config: Hydra config dict for the encoder (Evo2, ESM3, or Orthrus).
        modality: Which modality to encode ('dna', 'rna', or 'protein').
        batch_size: Number of sequences to encode at once (for memory efficiency).
        normalize: If True, L2-normalize embeddings.
        save_path: Optional path to save embeddings as .pt file.
        n_components: Expected embedding dimension. Auto-detected if None.
        **kwargs: Passed to LatentModule (datamodule, init_seed, etc.)

    Example:
        >>> encoder = BatchEncoder(
        ...     encoder_config={'_target_': 'manylatents.dogma.encoders.Evo2Encoder'},
        ...     modality='dna',
        ...     batch_size=8,
        ...     save_path='embeddings/evo2.pt',
        ...     datamodule=clinvar_dm,
        ... )
        >>> embeddings = encoder.fit_transform(x)  # x is ignored
    """

    # Default embedding dimensions
    DEFAULT_DIMS = {
        "dna": 1920,  # Evo2 1B hidden_size
        "rna": 256,  # Orthrus 4-track
        "protein": 1536,  # ESM3
    }

    def __init__(
        self,
        encoder_config: Dict,
        modality: str,
        batch_size: int = 8,
        normalize: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        n_components: Optional[int] = None,
        channel: Optional[str] = None,
        **kwargs,
    ):
        if modality not in ("dna", "rna", "protein"):
            raise ValueError(f"modality must be 'dna', 'rna', or 'protein', got '{modality}'")

        # Set n_components from default if not specified
        if n_components is None:
            n_components = self.DEFAULT_DIMS.get(modality, 1536)

        super().__init__(n_components=n_components, **kwargs)

        self._encoder_config = encoder_config
        self._modality = modality
        self._channel = channel  # 'wt' or 'mut' for VariantDataModule
        self._batch_size = batch_size
        self._normalize = normalize
        self._save_path = Path(save_path) if save_path else None

        # Lazy-loaded encoder
        self._encoder = None

    def _ensure_encoder_loaded(self) -> None:
        """Lazy-load encoder using Hydra instantiate."""
        if self._encoder is None:
            logger.info(f"Loading encoder for {self._modality}...")
            self._encoder = hydra.utils.instantiate(self._encoder_config)
            logger.info(f"Encoder loaded: {type(self._encoder).__name__}")

    def fit(self, x: Tensor, y: Tensor = None) -> None:
        """No-op fit - encoder is pretrained.

        Args:
            x: Input tensor (ignored - sequences come from datamodule).
            y: Labels (ignored - pretrained encoder).
        """
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Encode all sequences for the target modality.

        Gets sequences from datamodule.get_sequences()[modality] and encodes
        each one, returning stacked embeddings.

        Args:
            x: Input tensor (ignored - sequences come from datamodule).

        Returns:
            Embeddings tensor of shape (N, embedding_dim) where N is the
            number of sequences.

        Raises:
            ValueError: If datamodule is not set or missing get_sequences().
        """
        self._ensure_encoder_loaded()

        if self.datamodule is None:
            raise ValueError(
                "BatchEncoder requires a datamodule with get_sequences(). "
                "Pass datamodule= when instantiating."
            )

        if not hasattr(self.datamodule, "get_sequences"):
            raise ValueError(
                f"Datamodule {type(self.datamodule).__name__} has no get_sequences() method. "
                "Use ClinVarDataModule or CentralDogmaDataModule."
            )

        # Get sequences - try channel first (VariantDataModule), then modality (ClinVarDataModule)
        all_sequences = self.datamodule.get_sequences()

        # VariantDataModule returns {"wt": [...], "mut": [...]}
        # ClinVarDataModule returns {"dna": [...], "rna": [...], "protein": [...]}
        if self._channel and self._channel in all_sequences:
            sequences = all_sequences[self._channel]
            key_used = self._channel
        else:
            sequences = all_sequences.get(self._modality, [])
            key_used = self._modality

        if not sequences:
            raise ValueError(
                f"No sequences found for key '{key_used}'. "
                f"Available keys: {list(all_sequences.keys())}"
            )

        # Handle both single sequence (CentralDogma) and list (ClinVar)
        if isinstance(sequences, str):
            sequences = [sequences]

        logger.info(f"Encoding {len(sequences)} {self._modality} sequences...")

        # Filter empty sequences, track their indices for zero-fill
        valid_indices = []
        valid_sequences = []
        for i, seq in enumerate(sequences):
            if seq:
                valid_indices.append(i)
                valid_sequences.append(seq)

        # Encode via the encoder's batched path (true GPU batching if supported)
        if valid_sequences:
            valid_result = self._encoder.encode_batch(
                valid_sequences,
                batch_size=self._batch_size,
                show_progress=True,
            )
        else:
            valid_result = torch.zeros(0, self.n_components)

        # Scatter valid results into full-size tensors (zero-fill gaps)
        full_result = self._scatter_into_full(valid_result, valid_indices, len(sequences))

        # Save and extract primary embedding
        if isinstance(full_result, dict):
            if self._save_path:
                self._save_multi_layer(full_result)
            # Return first layer as primary embedding for the pipeline
            embeddings = full_result[next(iter(full_result))]
        else:
            if self._save_path:
                self._save_embeddings(full_result)
            embeddings = full_result

        # Normalize if requested
        if self._normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        logger.info(f"Encoded {len(embeddings)} sequences -> shape {embeddings.shape}")

        return embeddings

    @staticmethod
    def _scatter_one(v: Tensor, valid_indices: List[int], total: int) -> Tensor:
        """Place valid results into a full-sized zero tensor."""
        full = torch.zeros(total, v.shape[1])
        for j, idx in enumerate(valid_indices):
            full[idx] = v[j]
        return full

    def _scatter_into_full(
        self,
        result: Union[Tensor, Dict[str, Tensor]],
        valid_indices: List[int],
        total: int,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Scatter valid results into full-sized tensors, zero-filling gaps."""
        if len(valid_indices) == total:
            return result
        if isinstance(result, dict):
            return {k: self._scatter_one(v, valid_indices, total) for k, v in result.items()}
        return self._scatter_one(result, valid_indices, total)

    def _build_save_dict(self, embeddings: Tensor) -> dict:
        """Build save dict with embeddings + optional metadata."""
        save_dict = {"embeddings": embeddings}
        if self.datamodule is not None:
            if hasattr(self.datamodule, "get_variant_ids"):
                save_dict["variant_ids"] = self.datamodule.get_variant_ids()
            if hasattr(self.datamodule, "get_labels"):
                save_dict["labels"] = self.datamodule.get_labels()
        return save_dict

    def _save_multi_layer(self, embeddings_dict: Dict[str, Tensor]) -> None:
        """Save each layer's embeddings as a separate .pt file.

        For save_path='results/wt_dna.pt' and layer 'blocks.19.mlp.l3',
        saves to 'results/wt_dna_blocks.19.pt'.
        """
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        for layer_name, embeddings in embeddings_dict.items():
            parts = layer_name.split(".")
            short_name = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else layer_name
            layer_path = self._save_path.with_name(
                f"{self._save_path.stem}_{short_name}{self._save_path.suffix}"
            )
            torch.save(self._build_save_dict(embeddings), layer_path)
            logger.info(f"Saved layer {layer_name} embeddings to {layer_path}")

    def _save_embeddings(self, embeddings: Tensor) -> None:
        """Save embeddings to disk."""
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._build_save_dict(embeddings), self._save_path)
        logger.info(f"Saved embeddings to {self._save_path}")

    def cleanup(self) -> None:
        """Release encoder reference to free GPU memory."""
        if self._encoder is not None:
            del self._encoder
            self._encoder = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __getstate__(self):
        """Make picklable by excluding encoder."""
        state = self.__dict__.copy()
        state["_encoder"] = None
        return state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"modality={self._modality}, "
            f"n_components={self.n_components}, "
            f"batch_size={self._batch_size})"
        )
