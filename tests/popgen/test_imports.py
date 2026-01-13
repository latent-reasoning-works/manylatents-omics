"""
Test that manylatents-omics modules can be imported correctly.
"""

import pytest


def test_omics_package_import():
    """Test that the omics package can be imported."""
    import manylatents.popgen
    assert manylatents.popgen.__version__ == "0.1.0"


def test_data_module_imports():
    """Test that all data classes can be imported."""
    from manylatents.popgen.data import (
        PlinkDataset,
        PrecomputedMixin,
        HGDPDataset,
        HGDPDataModule,
        AOUDataset,
        AOUDataModule,
        UKBBDataset,
        UKBBDataModule,
        MHIDataset,
        MHIDataModule,
    )
    
    # Verify they're all defined
    assert PlinkDataset is not None
    assert PrecomputedMixin is not None
    assert HGDPDataset is not None
    assert HGDPDataModule is not None
    assert AOUDataset is not None
    assert AOUDataModule is not None
    assert UKBBDataset is not None
    assert UKBBDataModule is not None
    assert MHIDataset is not None
    assert MHIDataModule is not None


def test_metrics_module_imports():
    """Test that all metrics can be imported."""
    from manylatents.popgen.metrics import (
        GeographicPreservation,
        AdmixturePreservation,
        AdmixturePreservationK,
        AdmixtureLaplacian,
        SampleId,
    )
    # GroundTruthPreservation moved to core manylatents.metrics.preservation
    from manylatents.metrics.preservation import GroundTruthPreservation

    # Verify they're all callable
    assert callable(GeographicPreservation)
    assert callable(AdmixturePreservation)
    assert callable(AdmixturePreservationK)
    assert callable(AdmixtureLaplacian)
    assert callable(GroundTruthPreservation)
    assert callable(SampleId)


def test_namespace_package_structure():
    """Test that omics is properly set up as a namespace package."""
    import manylatents.popgen
    
    # Check that expected modules exist
    assert hasattr(manylatents.popgen, 'data')
    assert hasattr(manylatents.popgen, 'metrics')
    assert hasattr(manylatents.popgen, 'callbacks')
    assert hasattr(manylatents.popgen, 'utils')
    
    # Check __all__ exports
    assert 'data' in manylatents.popgen.__all__
    assert 'metrics' in manylatents.popgen.__all__


def test_core_manylatents_accessible():
    """Test that core manylatents can still be imported when omics is installed."""
    # This verifies namespace packages don't break core imports
    from manylatents.data.synthetic_dataset import SwissRoll
    from manylatents.data.test_data import TestDataset
    import manylatents.metrics.trustworthiness as trustworthiness_module
    
    assert SwissRoll is not None
    assert TestDataset is not None
    assert trustworthiness_module is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
