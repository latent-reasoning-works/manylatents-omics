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
    >>> print(embeddings.shape)  # (N, 2048) where N = number of variants

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
from tqdm import tqdm

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
        "dna": 2048,  # Evo2
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

    def fit(self, x: Tensor) -> None:
        """No-op fit - encoder is pretrained.

        Args:
            x: Input tensor (ignored - sequences come from datamodule).
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

        # Get sequences for this modality
        all_sequences = self.datamodule.get_sequences()
        sequences = all_sequences.get(self._modality, [])

        if not sequences:
            raise ValueError(f"No sequences found for modality '{self._modality}'")

        # Handle both single sequence (CentralDogma) and list (ClinVar)
        if isinstance(sequences, str):
            sequences = [sequences]

        logger.info(f"Encoding {len(sequences)} {self._modality} sequences...")

        # Encode in batches for memory efficiency
        all_embeddings = []
        for i in tqdm(range(0, len(sequences), self._batch_size), desc=f"Encoding {self._modality}"):
            batch = sequences[i : i + self._batch_size]

            # Encode batch (single sequence at a time for most encoders)
            batch_embeddings = []
            for seq in batch:
                if not seq:  # Skip empty sequences
                    # Return zeros for missing sequences
                    batch_embeddings.append(torch.zeros(self.n_components))
                    continue

                emb = self._encoder.encode(seq)
                if emb.dim() == 1:
                    emb = emb.squeeze()  # Remove extra dims
                batch_embeddings.append(emb.cpu())

            all_embeddings.extend(batch_embeddings)

        # Stack into tensor
        embeddings = torch.stack(all_embeddings, dim=0)

        # Normalize if requested
        if self._normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        logger.info(f"Encoded {len(embeddings)} sequences -> shape {embeddings.shape}")

        # Save if path specified
        if self._save_path:
            self._save_embeddings(embeddings)

        return embeddings

    def _save_embeddings(self, embeddings: Tensor) -> None:
        """Save embeddings to disk."""
        self._save_path.parent.mkdir(parents=True, exist_ok=True)

        # Also save variant IDs if available
        save_dict = {"embeddings": embeddings}

        if hasattr(self.datamodule, "get_variant_ids"):
            save_dict["variant_ids"] = self.datamodule.get_variant_ids()

        if hasattr(self.datamodule, "get_labels"):
            save_dict["labels"] = self.datamodule.get_labels()

        torch.save(save_dict, self._save_path)
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
