"""Algorithm adapters for central dogma foundation models.

Provides LatentModule implementations for central dogma analysis:
    - CentralDogmaFusion: Concatenate DNA, RNA, and Protein embeddings
    - BatchEncoder: Encode batches of sequences with a single encoder

Foundation model encoders can also be used directly via inference mode:
    algorithms:
      inference:
        _target_: manylatents.dogma.encoders.ESM3Encoder

Example (fusion):
    >>> from manylatents.dogma.algorithms import CentralDogmaFusion
    >>> fusion = CentralDogmaFusion(
    ...     evo2_config={'_target_': 'manylatents.dogma.encoders.Evo2Encoder'},
    ...     esm3_config={'_target_': 'manylatents.dogma.encoders.ESM3Encoder'},
    ...     datamodule=datamodule,
    ... )
    >>> embeddings = fusion.fit_transform(dummy_tensor)

Example (batch encoding):
    >>> from manylatents.dogma.algorithms import BatchEncoder
    >>> encoder = BatchEncoder(
    ...     encoder_config={'_target_': 'manylatents.dogma.encoders.Evo2Encoder'},
    ...     modality='dna',
    ...     datamodule=clinvar_dm,
    ... )
    >>> embeddings = encoder.fit_transform(dummy_tensor)
"""

from .fusion import CentralDogmaFusion, CentralDogmaEmbeddings
from .batch_encoder import BatchEncoder

__all__ = ["CentralDogmaFusion", "CentralDogmaEmbeddings", "BatchEncoder"]
