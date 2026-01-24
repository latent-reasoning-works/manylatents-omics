"""Central dogma fusion: concatenate DNA, RNA, Protein embeddings.

Fusion algorithm that concatenates embeddings from all 3 central dogma
encoders (Evo2 for DNA, Orthrus for RNA, ESM3 for Protein). Follows the
central dogma order: DNA -> RNA -> Protein.

Output dimension: 2048 (DNA) + 256 (RNA) + 1536 (Protein) = 3840

Example:
    >>> fusion = CentralDogmaFusion(
    ...     evo2_config={'_target_': 'manylatents.dogma.encoders.Evo2Encoder'},
    ...     orthrus_config={'_target_': 'manylatents.dogma.encoders.OrthrusEncoder'},
    ...     esm3_config={'_target_': 'manylatents.dogma.encoders.ESM3Encoder'},
    ...     datamodule=central_dogma_datamodule,
    ... )
    >>> fusion.fit(dummy_tensor)  # no-op
    >>> embeddings = fusion.transform(dummy_tensor)  # (batch, 3840)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import hydra
import torch
from torch import Tensor

from manylatents.algorithms.latent.latent_module_base import LatentModule


@dataclass
class CentralDogmaEmbeddings:
    """Container for embeddings from all three central dogma modalities.

    Attributes:
        dna: Evo2 embeddings, shape (batch, 2048)
        rna: Orthrus embeddings, shape (batch, 256)
        protein: ESM3 embeddings, shape (batch, 1536)
    """

    dna: Tensor  # (batch, 2048)
    rna: Tensor  # (batch, 256)
    protein: Tensor  # (batch, 1536)

    @property
    def total_dim(self) -> int:
        """Total embedding dimension across all modalities."""
        return self.dna.shape[-1] + self.rna.shape[-1] + self.protein.shape[-1]

    def concatenate(self) -> Tensor:
        """Concatenate embeddings in central dogma order: DNA -> RNA -> Protein."""
        return torch.cat([self.dna, self.rna, self.protein], dim=-1)


class CentralDogmaFusion(LatentModule):
    """Concatenate embeddings from DNA, RNA, and Protein foundation models.

    This fusion algorithm encodes sequences using all three central dogma
    foundation models and concatenates the resulting embeddings:
        - DNA: Evo2 (2048-dim)
        - RNA: Orthrus (256-dim for 4-track, 512-dim for 6-track)
        - Protein: ESM3 (1536-dim)

    Follows central dogma order: DNA -> RNA -> Protein
    Default output dimension: 2048 + 256 + 1536 = 3840

    Args:
        evo2_config: Hydra config dict for Evo2Encoder.
        orthrus_config: Hydra config dict for OrthrusEncoder.
        esm3_config: Hydra config dict for ESM3Encoder.
        normalize: If True, L2-normalize each modality before concatenation.
        n_components: Expected output dimension (default 3840).
        **kwargs: Passed to LatentModule (datamodule, init_seed, etc.)

    Example:
        >>> fusion = CentralDogmaFusion(
        ...     evo2_config={'_target_': 'manylatents.dogma.encoders.Evo2Encoder'},
        ...     orthrus_config={'_target_': 'manylatents.dogma.encoders.OrthrusEncoder'},
        ...     esm3_config={'_target_': 'manylatents.dogma.encoders.ESM3Encoder'},
        ...     datamodule=datamodule,
        ... )
        >>> embeddings = fusion.fit_transform(dummy_tensor)
        >>> print(embeddings.shape)  # (1, 3840)
    """

    # Default embedding dimensions for each encoder
    DEFAULT_DIMS = {
        "evo2": 2048,  # evo2_1b_base
        "orthrus": 256,  # 4-track base model
        "esm3": 1536,  # esm3-sm-open
    }

    def __init__(
        self,
        evo2_config: Dict,
        orthrus_config: Dict,
        esm3_config: Dict,
        normalize: bool = False,
        n_components: int = 3840,
        **kwargs,
    ):
        super().__init__(n_components=n_components, **kwargs)
        self._evo2_config = evo2_config
        self._orthrus_config = orthrus_config
        self._esm3_config = esm3_config
        self._normalize = normalize

        # Lazy-loaded encoders (loaded on first transform call)
        self._evo2 = None
        self._orthrus = None
        self._esm3 = None

    def _ensure_encoders_loaded(self) -> None:
        """Lazy-load encoders using Hydra instantiate."""
        if self._evo2 is None:
            self._evo2 = hydra.utils.instantiate(self._evo2_config)
        if self._orthrus is None:
            self._orthrus = hydra.utils.instantiate(self._orthrus_config)
        if self._esm3 is None:
            self._esm3 = hydra.utils.instantiate(self._esm3_config)

    def fit(self, x: Tensor) -> None:
        """No-op fit - all encoders are pretrained.

        Args:
            x: Input tensor (ignored - sequences come from datamodule).
        """
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Encode sequences and concatenate embeddings.

        Gets sequences from the datamodule's get_sequences() method,
        encodes each modality, and concatenates in central dogma order.

        Args:
            x: Input tensor (ignored - sequences come from datamodule).

        Returns:
            Concatenated embeddings of shape (batch, n_components).
            Default: (batch, 3840) = DNA(2048) + RNA(256) + Protein(1536)

        Raises:
            ValueError: If datamodule is not set or missing get_sequences().
        """
        self._ensure_encoders_loaded()

        if self.datamodule is None:
            raise ValueError(
                "CentralDogmaFusion requires a datamodule with get_sequences(). "
                "Pass datamodule= when instantiating or use experiment config."
            )

        if not hasattr(self.datamodule, "get_sequences"):
            raise ValueError(
                f"Datamodule {type(self.datamodule).__name__} has no get_sequences() method. "
                "Use CentralDogmaDataModule for fusion experiments."
            )

        # Get sequences from datamodule (dict with dna, rna, protein keys)
        sequences = self.datamodule.get_sequences()

        # Encode each modality
        dna_emb = self._evo2.encode(sequences["dna"])
        rna_emb = self._orthrus.encode(sequences["rna"])
        protein_emb = self._esm3.encode(sequences["protein"])

        # Ensure batch dimension
        if dna_emb.dim() == 1:
            dna_emb = dna_emb.unsqueeze(0)
        if rna_emb.dim() == 1:
            rna_emb = rna_emb.unsqueeze(0)
        if protein_emb.dim() == 1:
            protein_emb = protein_emb.unsqueeze(0)

        # Optional L2 normalization per modality
        if self._normalize:
            dna_emb = torch.nn.functional.normalize(dna_emb, p=2, dim=-1)
            rna_emb = torch.nn.functional.normalize(rna_emb, p=2, dim=-1)
            protein_emb = torch.nn.functional.normalize(protein_emb, p=2, dim=-1)

        # Concatenate in central dogma order: DNA -> RNA -> Protein
        fused = torch.cat([dna_emb, rna_emb, protein_emb], dim=-1)

        return fused

    def get_embeddings(self, x: Tensor) -> CentralDogmaEmbeddings:
        """Get individual embeddings before concatenation.

        Useful for analysis or visualization of individual modality contributions.

        Args:
            x: Input tensor (ignored - sequences come from datamodule).

        Returns:
            CentralDogmaEmbeddings dataclass with dna, rna, protein tensors.
        """
        self._ensure_encoders_loaded()

        if self.datamodule is None:
            raise ValueError("Datamodule not set. Pass datamodule= when instantiating.")

        sequences = self.datamodule.get_sequences()

        dna_emb = self._evo2.encode(sequences["dna"])
        rna_emb = self._orthrus.encode(sequences["rna"])
        protein_emb = self._esm3.encode(sequences["protein"])

        # Ensure batch dimension
        if dna_emb.dim() == 1:
            dna_emb = dna_emb.unsqueeze(0)
        if rna_emb.dim() == 1:
            rna_emb = rna_emb.unsqueeze(0)
        if protein_emb.dim() == 1:
            protein_emb = protein_emb.unsqueeze(0)

        if self._normalize:
            dna_emb = torch.nn.functional.normalize(dna_emb, p=2, dim=-1)
            rna_emb = torch.nn.functional.normalize(rna_emb, p=2, dim=-1)
            protein_emb = torch.nn.functional.normalize(protein_emb, p=2, dim=-1)

        return CentralDogmaEmbeddings(dna=dna_emb, rna=rna_emb, protein=protein_emb)

    @property
    def embedding_dims(self) -> Dict[str, int]:
        """Return embedding dimensions for each modality."""
        dims = dict(self.DEFAULT_DIMS)
        # Update with actual dims if encoders are loaded
        if self._evo2 is not None:
            dims["evo2"] = self._evo2.embedding_dim
        if self._orthrus is not None:
            dims["orthrus"] = self._orthrus.embedding_dim
        if self._esm3 is not None:
            dims["esm3"] = self._esm3.embedding_dim
        return dims

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_components={self.n_components}, "
            f"normalize={self._normalize})"
        )
