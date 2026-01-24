"""Auto-discover manylatents-omics configs when package is installed.

Hydra automatically finds this plugin via the hydra_plugins namespace package.
No registration needed - just install manylatents-omics and configs are available.

This enables standalone use without setting HYDRA_SEARCH_PACKAGES:
    python -m manylatents.main +experiment=central_dogma_fusion

Note: This plugin also adds core manylatents configs since Hydra 1.3 doesn't
reliably discover entry-point plugins when namespace plugins are present.
"""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class OmicsSearchPathPlugin(SearchPathPlugin):
    """Automatically add omics and core manylatents config packages to Hydra's search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Add manylatents-omics and core manylatents config packages to Hydra's search path.

        Called automatically by Hydra during initialization.

        Search path order (first = highest priority for same-name configs):
        1. manylatents.dogma.configs (omics dogma - prepended)
        2. manylatents.configs (core manylatents - appended)
        3. Other omics submodule configs (popgen, singlecell - appended)
        """
        # Add core manylatents configs first (will be lower priority after prepend below)
        # This ensures base configs (algorithms, data, experiment, etc.) are available
        try:
            import manylatents.configs
            search_path.append(
                provider="manylatents",
                path="pkg://manylatents.configs",
            )
        except ImportError:
            pass

        # Add dogma configs with higher priority (prepend)
        # Dogma configs can override/extend core manylatents configs
        search_path.prepend(
            provider="manylatents-omics",
            path="pkg://manylatents.dogma.configs",
        )

        # Add popgen configs if available
        try:
            import manylatents.popgen.configs
            search_path.append(
                provider="manylatents-omics",
                path="pkg://manylatents.popgen.configs",
            )
        except ImportError:
            pass

        # Add singlecell configs if available
        try:
            import manylatents.singlecell.configs
            search_path.append(
                provider="manylatents-omics",
                path="pkg://manylatents.singlecell.configs",
            )
        except ImportError:
            pass
