"""Resolve the omics data root in an install-location-independent way.

The ``${omics_data:}`` OmegaConf resolver (see ``omics_plugin.py``) and the
dataset download scripts both read from here so they always agree on where
``.h5ad`` files live — regardless of whether the package is an editable source
checkout or a non-editable wheel in ``site-packages``.

Resolution order:

1. ``$MANYLATENTS_DATA`` — explicit override, wins unconditionally.
2. ``<repo>/data`` — when running from a source checkout (the repo root, two
   levels up from this file, still contains ``pyproject.toml``).
3. ``$XDG_CACHE_HOME/manylatents/data`` (falling back to
   ``~/.cache/manylatents/data``) — for installed packages, where the old
   ``parents[1]/data`` pointed *inside* ``site-packages`` and never existed.
"""

import os
from pathlib import Path

# <repo root> in a source checkout, or <site-packages> once installed.
_PKG_PARENT = Path(__file__).resolve().parents[1]


def omics_data_root() -> Path:
    """Return the base directory that holds omics datasets (e.g. ``single_cell/``)."""
    env = os.environ.get("MANYLATENTS_DATA")
    if env:
        return Path(env).expanduser()

    # Source checkout: the repo root carries pyproject.toml; site-packages does not.
    if (_PKG_PARENT / "pyproject.toml").is_file():
        return _PKG_PARENT / "data"

    # Installed wheel: use a writable per-user cache dir instead of site-packages.
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return base / "manylatents" / "data"
