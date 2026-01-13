"""Tests for PlotAdmixture callback."""
import pytest
from manylatents.popgen.callbacks import PlotAdmixture


def test_plot_admixture_import():
    """Test that PlotAdmixture can be imported."""
    assert PlotAdmixture is not None


def test_plot_admixture_instantiation():
    """Test that PlotAdmixture can be instantiated."""
    callback = PlotAdmixture(
        save_dir="/tmp",
        experiment_name="test",
        admixture_K=5
    )
    assert callback.admixture_K == 5
    assert callback.experiment_name == "test"


def test_plot_admixture_k_validation():
    """Test that PlotAdmixture validates K <= 10."""
    with pytest.raises(ValueError, match="admixture_K must be <= 10"):
        PlotAdmixture(admixture_K=11)
