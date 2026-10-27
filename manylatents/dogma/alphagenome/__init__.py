"""JAX-native AlphaGenome: local weights, per-layer site states, and their features.

Submodules import JAX and Haiku only when used; ``encoder`` alone imports
torch (through ``LatentModule``).
"""

__all__ = ["model", "taps", "site_states", "encoder"]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
