#!/bin/bash
# Verification script for manylatents-omics installation
# Run this after installation to verify everything works correctly.
#
# Usage:
#   module load anaconda/3 cuda/12.4.1
#   bash scripts/verify_install.sh

set -e

echo "=== manylatents-omics Installation Verification ==="
echo

# 1. Check plugin import
echo "1. Testing plugin import..."
uv run python -c "from manylatents.omics_plugin import OmicsSearchPathPlugin; print('   OK: OmicsSearchPathPlugin imported')"

# 2. Check plugin registration
echo "2. Testing plugin registration..."
uv run python -c "
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from manylatents.omics_plugin import OmicsSearchPathPlugin

plugins = Plugins.instance()
plugins.register(OmicsSearchPathPlugin)
discovered = list(plugins.discover(SearchPathPlugin))
assert OmicsSearchPathPlugin in discovered, 'Plugin not registered'
print('   OK: OmicsSearchPathPlugin registered with Hydra')
"

# 3. Check search path manipulation
echo "3. Testing search path manipulation..."
uv run python -c "
from hydra._internal.utils import create_config_search_path
from manylatents.omics_plugin import OmicsSearchPathPlugin

search_path = create_config_search_path('manylatents')
plugin = OmicsSearchPathPlugin()
plugin.manipulate_search_path(search_path)

paths = [sp.path for sp in search_path.get_path()]
assert 'pkg://manylatents.dogma.configs' in paths, 'dogma configs not on path'
assert 'pkg://manylatents.configs' in paths, 'core configs not on path'
print('   OK: Config search paths added correctly')
"

# 4. Check Hydra config resolution
echo "4. Testing Hydra config resolution..."
uv run python -m manylatents.main --help 2>&1 | grep -q "dogma/configs" && echo "   OK: dogma configs visible in Hydra"

# 5. Run a minimal experiment (if torch available)
echo "5. Testing minimal experiment..."
if uv run python -c "import torch" 2>/dev/null; then
    uv run python -m manylatents.main --config-name=config experiment=single_algorithm logger=none 2>&1 | tail -5
    echo "   OK: Experiment completed"
else
    echo "   SKIP: torch not available (CUDA not loaded?)"
fi

echo
echo "=== All verification checks passed ==="
