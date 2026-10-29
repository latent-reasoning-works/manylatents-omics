"""The omics portion of the manylatents namespace must not hide core's top-level API.

manylatents/__init__.py here must stay byte-identical to the core package's: in a pkgutil-style namespace,
whichever portion's __init__.py comes first on sys.path is the one that runs.
"""
import manylatents


def test_core_lazy_api_survives_omics_install():
    from manylatents import evaluate_metrics  # noqa: F401  (lazy import defined in core's __init__)


def test_version_is_core_version():
    from importlib.metadata import version
    assert manylatents.__version__ == version("manylatents")


def test_omics_subpackages_still_resolve():
    import manylatents.singlecell  # noqa: F401
