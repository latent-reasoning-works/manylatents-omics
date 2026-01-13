"""
Test that manylatents-dogma modules can be imported correctly.
"""

import pytest


def test_dogma_package_import():
    """Test that the dogma package can be imported."""
    import manylatents.dogma
    assert manylatents.dogma.__version__ == "0.1.0"


def test_encoder_module_imports():
    """Test that all encoder classes can be imported."""
    from manylatents.dogma.encoders import (
        ESM3Encoder,
        OrthrusEncoder,
        Evo2Encoder,
    )

    # Verify they're all defined
    assert ESM3Encoder is not None
    assert OrthrusEncoder is not None
    assert Evo2Encoder is not None


def test_namespace_package_structure():
    """Test that dogma is properly set up as a namespace package."""
    import manylatents.dogma

    # Check that expected modules exist
    assert hasattr(manylatents.dogma, 'encoders')

    # Check __all__ exports
    assert 'encoders' in manylatents.dogma.__all__


def test_encoder_modalities():
    """Test that encoders have correct modalities defined."""
    from manylatents.dogma.encoders import (
        ESM3Encoder,
        OrthrusEncoder,
        Evo2Encoder,
    )

    # Check modality without instantiating (would require model loading)
    # These are class attributes
    assert ESM3Encoder.DEFAULT_WEIGHTS is not None
    assert OrthrusEncoder.MODELS is not None
    assert Evo2Encoder.MODELS is not None


def test_foundation_encoder_base_accessible():
    """Test that FoundationEncoder base class can be imported."""
    from manylatents.algorithms.encoder import FoundationEncoder

    assert FoundationEncoder is not None


def test_core_manylatents_still_accessible():
    """Test that core manylatents can still be imported with dogma installed."""
    from manylatents.data.synthetic_dataset import SwissRoll
    from manylatents.data.test_data import TestDataset
    import manylatents.metrics.trustworthiness as trustworthiness_module

    assert SwissRoll is not None
    assert TestDataset is not None
    assert trustworthiness_module is not None


def test_popgen_and_dogma_coexist():
    """Test that both popgen and dogma can be imported together."""
    import manylatents.popgen
    import manylatents.dogma

    assert manylatents.popgen.__version__ == "0.1.0"
    assert manylatents.dogma.__version__ == "0.1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
