"""
manylatents-dogma: Central dogma foundation model encoders for manylatents.

Submodules are imported lazily to avoid circular imports during Hydra
config scanning (configs/ is a sibling package to data/, encoders/, etc.).
"""

__version__ = "0.1.0"

__all__ = ["encoders", "algorithms", "data"]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
