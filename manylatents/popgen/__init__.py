"""
manylatents-popgen: Population genetics extension for manylatents.

This package extends manylatents with population genetics-specific datasets,
metrics, and visualizations. Includes support for:
    - PLINK format genotype data (HGDP, UK Biobank, All of Us)
    - Geographic and admixture preservation metrics
    - Population structure visualizations
"""

# Export popgen-specific modules
from . import data, metrics, callbacks, utils

__version__ = "0.1.0"

__all__ = ["data", "metrics", "callbacks", "utils"]
