"""Auto-discover manylatents-omics configs when package is installed.

This plugin is registered via auto-registration in manylatents.dogma.__init__
when any omics module is imported. This works reliably regardless of how the
package is installed (editable, git, wheel).

Note: Entry-point registration in pyproject.toml is also present but Hydra 1.3
doesn't actually use it for SearchPathPlugin discovery - it only scans the
hydra_plugins namespace package.
"""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class OmicsSearchPathPlugin(SearchPathPlugin):
    """Automatically add omics and core manylatents config packages to Hydra's search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Add omics config packages to Hydra's search path.

        Core manylatents configs are handled by ManylatentsSearchPathPlugin
        (registered in manylatents.configs.__init__).
        """
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
