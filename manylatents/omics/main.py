"""Entry point for manylatents with omics configs.

This module registers the OmicsSearchPathPlugin BEFORE Hydra initializes,
ensuring omics configs (dogma, popgen, singlecell) are available.

Usage:
    python -m manylatents.omics.main experiment=central_dogma_fusion
    python -m manylatents.omics.main data=hgdp algorithm=pca

This is equivalent to running `python -m manylatents.main` but with omics
configs automatically on the Hydra search path.
"""

# Register omics SearchPathPlugin BEFORE importing manylatents.main
# This ensures configs are available when @hydra.main() initializes
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from manylatents.omics_plugin import OmicsSearchPathPlugin

plugins = Plugins.instance()
existing = list(plugins.discover(SearchPathPlugin))
if OmicsSearchPathPlugin not in existing:
    plugins.register(OmicsSearchPathPlugin)

# Now import and run the main function
from manylatents.main import main

if __name__ == "__main__":
    main()
