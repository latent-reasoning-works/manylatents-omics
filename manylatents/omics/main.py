"""Entry point for manylatents with omics configs.

Importing omics_plugin registers the OmicsSearchPathPlugin and the
omics_data resolver before Hydra initializes.

Usage:
    python -m manylatents.omics.main experiment=central_dogma_fusion
    python -m manylatents.omics.main data=embryoid_body algorithm=pca
"""

import manylatents.omics_plugin  # noqa: F401 — registers plugin + resolver on import

from manylatents.main import main

if __name__ == "__main__":
    main()
