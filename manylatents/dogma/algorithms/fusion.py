"""Central dogma fusion: concatenate DNA, RNA, Protein embeddings.

Fusion algorithm that concatenates embeddings from central dogma foundation
models. Supports flexible modality selection:
    - DNA: Evo2 (1920-dim)
    - RNA: Orthrus (256-dim) - currently blocked on mamba-ssm version conflict
    - Protein: ESM3 (1536-dim)

Each encoder config is optional. Omit a config (set to None) to skip that modality.
Embeddings are concatenated in central dogma order: DNA -> RNA -> Protein.

Default output dimension (all 3): 1920 + 256 + 1536 = 3712
DNA + Protein only: 1920 + 1536 = 3456

Example:
    >>> # Full fusion (requires mamba-ssm compatibility)
    >>> fusion = CentralDogmaFusion(
    ...     evo2_config={'_target_': 'manylatents.dogma.encoders.Evo2Encoder'},
    ...     orthrus_config={'_target_': 'manylatents.dogma.encoders.OrthrusEncoder'},
    ...     esm3_config={'_target_': 'manylatents.dogma.encoders.ESM3Encoder'},
    ...     datamodule=central_dogma_datamodule,
    ... )

    >>> # DNA + Protein only (works now)
    >>> fusion = CentralDogmaFusion(
    ...     evo2_config={'_target_': 'manylatents.dogma.encoders.Evo2Encoder'},
    ...     orthrus_config=None,  # Skip RNA
    ...     esm3_config={'_target_': 'manylatents.dogma.encoders.ESM3Encoder'},
    ...     n_components=3456,  # 1920 + 1536
    ...     datamodule=datamodule,
    ... )
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
        dna: Evo2 embeddings, shape (batch, 1920)
        rna: Orthrus embeddings, shape (batch, 256)
        protein: ESM3 embeddings, shape (batch, 1536)
    """

    dna: Tensor  # (batch, 1920)
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

    This fusion algorithm encodes sequences using central dogma foundation
    models and concatenates the resulting embeddings. Each encoder is optional:
        - DNA: Evo2 (1920-dim)
        - RNA: Orthrus (256-dim for 4-track, 512-dim for 6-track)
        - Protein: ESM3 (1536-dim)

    At least one encoder must be provided. Embeddings are concatenated in
    central dogma order: DNA -> RNA -> Protein (skipping None modalities).

    Args:
        evo2_config: Hydra config dict for Evo2Encoder, or None to skip DNA.
        orthrus_config: Hydra config dict for OrthrusEncoder, or None to skip RNA.
        esm3_config: Hydra config dict for ESM3Encoder, or None to skip Protein.
        normalize: If True, L2-normalize each modality before concatenation.
        n_components: Expected output dimension. If None, computed from active encoders.
        **kwargs: Passed to LatentModule (datamodule, init_seed, etc.)

    Example:
        >>> # DNA + Protein fusion (skip Orthrus due to mamba-ssm conflict)
        >>> fusion = CentralDogmaFusion(
        ...     evo2_config={'_target_': 'manylatents.dogma.encoders.Evo2Encoder'},
        ...     orthrus_config=None,
        ...     esm3_config={'_target_': 'manylatents.dogma.encoders.ESM3Encoder'},
        ...     datamodule=datamodule,
        ... )
        >>> embeddings = fusion.fit_transform(dummy_tensor)
        >>> print(embeddings.shape)  # (1, 3584) = DNA(2048) + Protein(1536)
    """

    # Default embedding dimensions for each encoder
    DEFAULT_DIMS = {
        "evo2": 1920,  # evo2_1b_base
        "orthrus": 256,  # 4-track base model
        "esm3": 1536,  # esm3-sm-open
    }

    def __init__(
        self,
        evo2_config: Optional[Dict] = None,
        orthrus_config: Optional[Dict] = None,
        esm3_config: Optional[Dict] = None,
        normalize: bool = False,
        n_components: Optional[int] = None,
        **kwargs,
    ):
        # Validate at least one encoder is provided
        if evo2_config is None and orthrus_config is None and esm3_config is None:
            raise ValueError("At least one encoder config must be provided")

        # Compute n_components from active encoders if not specified
        if n_components is None:
            n_components = 0
            if evo2_config is not None:
                n_components += self.DEFAULT_DIMS["evo2"]
            if orthrus_config is not None:
                n_components += self.DEFAULT_DIMS["orthrus"]
            if esm3_config is not None:
                n_components += self.DEFAULT_DIMS["esm3"]

        super().__init__(n_components=n_components, **kwargs)
        self._evo2_config = evo2_config
        self._orthrus_config = orthrus_config
        self._esm3_config = esm3_config
        self._normalize = normalize

        # Track which modalities are active
        self._active_modalities = []
        if evo2_config is not None:
            self._active_modalities.append("dna")
        if orthrus_config is not None:
            self._active_modalities.append("rna")
        if esm3_config is not None:
            self._active_modalities.append("protein")

        # Lazy-loaded encoders (loaded on first transform call)
        self._evo2 = None
        self._orthrus = None
        self._esm3 = None

    def _ensure_encoders_loaded(self) -> None:
        """Lazy-load configured encoders using Hydra instantiate."""
        if self._evo2_config is not None and self._evo2 is None:
            self._evo2 = hydra.utils.instantiate(self._evo2_config)
        if self._orthrus_config is not None and self._orthrus is None:
            self._orthrus = hydra.utils.instantiate(self._orthrus_config)
        if self._esm3_config is not None and self._esm3 is None:
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
        encodes each active modality, and concatenates in central dogma order.

        Args:
            x: Input tensor (ignored - sequences come from datamodule).

        Returns:
            Concatenated embeddings of shape (batch, n_components).
            Dimension depends on active encoders.

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

        # Collect embeddings for active modalities in central dogma order
        embeddings = []

        # DNA (Evo2)
        if self._evo2 is not None:
            dna_emb = self._evo2.encode(sequences["dna"])
            if dna_emb.dim() == 1:
                dna_emb = dna_emb.unsqueeze(0)
            if self._normalize:
                dna_emb = torch.nn.functional.normalize(dna_emb, p=2, dim=-1)
            embeddings.append(dna_emb)

        # RNA (Orthrus)
        if self._orthrus is not None:
            rna_emb = self._orthrus.encode(sequences["rna"])
            if rna_emb.dim() == 1:
                rna_emb = rna_emb.unsqueeze(0)
            if self._normalize:
                rna_emb = torch.nn.functional.normalize(rna_emb, p=2, dim=-1)
            embeddings.append(rna_emb)

        # Protein (ESM3)
        if self._esm3 is not None:
            protein_emb = self._esm3.encode(sequences["protein"])
            if protein_emb.dim() == 1:
                protein_emb = protein_emb.unsqueeze(0)
            if self._normalize:
                protein_emb = torch.nn.functional.normalize(protein_emb, p=2, dim=-1)
            embeddings.append(protein_emb)

        # Concatenate in central dogma order
        fused = torch.cat(embeddings, dim=-1)

        return fused

    def get_embeddings(self, x: Tensor) -> Dict[str, Tensor]:
        """Get individual embeddings before concatenation.

        Useful for analysis or visualization of individual modality contributions.

        Args:
            x: Input tensor (ignored - sequences come from datamodule).

        Returns:
            Dict mapping modality names to embedding tensors.
            Only includes active modalities.
        """
        self._ensure_encoders_loaded()

        if self.datamodule is None:
            raise ValueError("Datamodule not set. Pass datamodule= when instantiating.")

        sequences = self.datamodule.get_sequences()
        embeddings = {}

        if self._evo2 is not None:
            dna_emb = self._evo2.encode(sequences["dna"])
            if dna_emb.dim() == 1:
                dna_emb = dna_emb.unsqueeze(0)
            if self._normalize:
                dna_emb = torch.nn.functional.normalize(dna_emb, p=2, dim=-1)
            embeddings["dna"] = dna_emb

        if self._orthrus is not None:
            rna_emb = self._orthrus.encode(sequences["rna"])
            if rna_emb.dim() == 1:
                rna_emb = rna_emb.unsqueeze(0)
            if self._normalize:
                rna_emb = torch.nn.functional.normalize(rna_emb, p=2, dim=-1)
            embeddings["rna"] = rna_emb

        if self._esm3 is not None:
            protein_emb = self._esm3.encode(sequences["protein"])
            if protein_emb.dim() == 1:
                protein_emb = protein_emb.unsqueeze(0)
            if self._normalize:
                protein_emb = torch.nn.functional.normalize(protein_emb, p=2, dim=-1)
            embeddings["protein"] = protein_emb

        return embeddings

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

    def cleanup(self) -> None:
        """Release encoder references to free GPU memory and enable pickling.

        Call this after encoding is complete to allow the result to be pickled
        by submitit or other job runners.
        """
        if self._evo2 is not None:
            del self._evo2
            self._evo2 = None
        if self._orthrus is not None:
            del self._orthrus
            self._orthrus = None
        if self._esm3 is not None:
            del self._esm3
            self._esm3 = None

        # Clear CUDA cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __getstate__(self):
        """Make the object picklable by excluding encoder references."""
        state = self.__dict__.copy()
        # Replace unpicklable encoder instances with None
        state['_evo2'] = None
        state['_orthrus'] = None
        state['_esm3'] = None
        return state

    def __repr__(self) -> str:
        modalities = "+".join(m.upper() for m in self._active_modalities)
        return (
            f"{self.__class__.__name__}("
            f"modalities={modalities}, "
            f"n_components={self.n_components}, "
            f"normalize={self._normalize})"
        )
