"""
manylatents-omics: Genetics and genomics extension for manylatents

This package extends manylatents with genetics-specific datasets, metrics,
and visualizations for population genetics and genomics analysis.
"""

# Export omics-specific modules
from . import data, metrics, callbacks, utils

__version__ = "0.1.0"

__all__ = ["data", "metrics", "callbacks", "utils"]
