"""Genetics-specific callbacks for manylatents."""

__all__ = ["PlotAdmixture"]


def __getattr__(name):
    if name == "PlotAdmixture":
        from manylatents.popgen.callbacks.plot_admixture import PlotAdmixture
        return PlotAdmixture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
