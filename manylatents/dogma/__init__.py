"""
manylatents-dogma: Central dogma foundation model encoders for manylatents.

Submodules are imported lazily to avoid circular imports during Hydra
config scanning (configs/ is a sibling package to data/, encoders/, etc.).

``vep`` is the variant-effect API surface (parse_mutation, encode_variant,
compute_llr, etc.) that the workshop's Path A/B/C all converge on after
the 2.11 collapse. It composes with ESMEncoder.encode_with_logits.
"""

__version__ = "0.1.0"

__all__ = ["encoders", "algorithms", "data", "vep"]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
