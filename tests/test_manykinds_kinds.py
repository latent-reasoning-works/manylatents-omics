"""The singlecell extra installs the manykinds array kinds."""
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_singlecell_extra_requires_the_arrays_extra():
    assert ROOT.is_dir()
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert "manykinds[arrays]>=0.1.2" in extras["singlecell"]


def test_array_kinds_import():
    manykinds = pytest.importorskip("manykinds")
    assert manykinds.LabeledArray is not None
    assert manykinds.SparseGraph is not None
