"""Test that singlecell module imports correctly."""
import pytest


def test_singlecell_package_import():
    import manylatents.singlecell
    assert manylatents.singlecell is not None


def test_data_module_imports():
    from manylatents.singlecell.data import AnnDataset, AnnDataModule
    assert AnnDataset is not None
    assert AnnDataModule is not None


def test_analysis_module_imports():
    sc = pytest.importorskip("scanpy")
    from manylatents.singlecell.analysis import (
        ComplementSetAnalysis,
        DifferentialExpression,
        EmbeddingAudit,
    )
    assert callable(ComplementSetAnalysis)
    assert callable(DifferentialExpression)
