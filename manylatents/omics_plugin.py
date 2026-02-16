"""Auto-discover manylatents-omics configs and data paths.

Mirrors the manylatents pattern (configs/__init__.py): define plugin inline,
register on import. Entry-points don't work with Hydra 1.3, so we register
manually.

Also registers the ``omics_data`` OmegaConf resolver so singlecell/popgen
data configs can use ``${omics_data:}/single_cell/file.h5ad`` — resolved
automatically from the installed package location.
"""

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import OmegaConf

# Omics repo root — two levels up from this file (manylatents/omics_plugin.py → repo root)
_OMICS_ROOT = Path(__file__).resolve().parents[1]


# --- OmegaConf resolver: ${omics_data:} → <omics_repo>/data ---

if not OmegaConf.has_resolver("omics_data"):
    OmegaConf.register_new_resolver(
        "omics_data",
        lambda: str(_OMICS_ROOT / "data"),
        use_cache=True,
    )


# --- Hydra SearchPathPlugin ---

class OmicsSearchPathPlugin(SearchPathPlugin):
    """Add omics config packages (dogma, popgen, singlecell) to Hydra's search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.prepend(
            provider="manylatents-omics",
            path="pkg://manylatents.dogma.configs",
        )
        search_path.append(
            provider="manylatents-omics",
            path="pkg://manylatents.popgen.configs",
        )
        search_path.append(
            provider="manylatents-omics",
            path="pkg://manylatents.singlecell.configs",
        )


# Register on import (same pattern as ManylatentsSearchPathPlugin)
_plugins = Plugins.instance()
if OmicsSearchPathPlugin not in list(_plugins.discover(SearchPathPlugin)):
    _plugins.register(OmicsSearchPathPlugin)
