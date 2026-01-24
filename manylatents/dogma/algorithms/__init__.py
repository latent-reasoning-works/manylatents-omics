"""Algorithm adapters for central dogma foundation models.

Provides LatentModule implementations for central dogma analysis:
    - CentralDogmaFusion: Concatenate DNA, RNA, and Protein embeddings

Foundation model encoders can also be used directly via inference mode:
    algorithms:
      inference:
        _target_: manylatents.dogma.encoders.ESM3Encoder

Example:
    >>> from manylatents.dogma.algorithms import CentralDogmaFusion
    >>> fusion = CentralDogmaFusion(
    ...     evo2_config={'_target_': 'manylatents.dogma.encoders.Evo2Encoder'},
    ...     orthrus_config={'_target_': 'manylatents.dogma.encoders.OrthrusEncoder'},
    ...     esm3_config={'_target_': 'manylatents.dogma.encoders.ESM3Encoder'},
    ...     datamodule=datamodule,
    ... )
    >>> embeddings = fusion.fit_transform(dummy_tensor)
"""

from .fusion import CentralDogmaFusion, CentralDogmaEmbeddings

__all__ = ["CentralDogmaFusion", "CentralDogmaEmbeddings"]
