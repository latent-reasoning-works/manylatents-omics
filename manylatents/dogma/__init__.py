"""
manylatents-dogma: Central dogma foundation model encoders for manylatents.

This package provides pretrained foundation model encoders for the three
levels of the central dogma:

    - DNA: Evo2 (StripedHyena2 architecture, 1B/7B/40B params)
    - RNA: Orthrus (Mamba-based, 4-track/6-track variants)
    - Protein: ESM3 (1.4B params)

These encoders transform biological sequences into dense embeddings that
can be used for downstream analysis with manylatents dimensionality
reduction algorithms.

Example:
    >>> from manylatents.dogma.encoders import ESM3Encoder, OrthrusEncoder, Evo2Encoder
    >>>
    >>> # Encode the same biological information across modalities
    >>> protein_emb = ESM3Encoder().encode("MKFGVRA")
    >>> rna_emb = OrthrusEncoder().encode("AUGAAGUUUGGCGUCCGUGCCUGA")
    >>> dna_emb = Evo2Encoder().encode("ATGAAGTTTGGCGTCCGTGCCTGA")

    >>> # Fusion: concatenate all three modalities
    >>> from manylatents.dogma.algorithms import CentralDogmaFusion
    >>> from manylatents.dogma.data import CentralDogmaDataModule
"""

__version__ = "0.1.0"

# Auto-register Hydra SearchPathPlugin when dogma is imported
# This ensures omics configs are available regardless of how the package is installed
# (editable, git, wheel). Hydra's namespace-based plugin discovery doesn't work reliably
# for editable installs from another project.
def _register_hydra_plugin():
    """Register OmicsSearchPathPlugin with Hydra if not already registered."""
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin
    from manylatents.omics_plugin import OmicsSearchPathPlugin

    plugins = Plugins.instance()
    # Check if already registered (avoid duplicate registration)
    existing = list(plugins.discover(SearchPathPlugin))
    if OmicsSearchPathPlugin not in existing:
        plugins.register(OmicsSearchPathPlugin)

_register_hydra_plugin()

# Direct submodule imports - required for Hydra's instantiate/get_class to work correctly
# The lazy __getattr__ pattern causes "maximum recursion depth exceeded" with Hydra
from . import encoders
from . import algorithms
from . import data

__all__ = ["encoders", "algorithms", "data"]
