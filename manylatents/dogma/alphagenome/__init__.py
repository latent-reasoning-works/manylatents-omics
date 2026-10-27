"""JAX-native AlphaGenome: local weights, per-layer site states, their features,
and light finetuning or steering of the states.

Submodules import JAX and Haiku only when used; only ``encoder`` and
``finetune`` import torch (through ``LatentModule``).
"""

__all__ = ["model", "taps", "site_states", "encoder", "interventions", "objectives", "train",
           "finetune"]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
