"""Tests for the install-location-independent omics data root resolution.

Regression coverage for issue #47: ``${omics_data:}`` resolved to a
non-existent ``site-packages/data`` path when the package was installed as a
non-editable wheel.
"""
from pathlib import Path

from omegaconf import OmegaConf

import manylatents._data_paths as dp
from manylatents._data_paths import omics_data_root


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("MANYLATENTS_DATA", str(tmp_path / "custom"))
    assert omics_data_root() == tmp_path / "custom"


def test_source_checkout_uses_repo_data(monkeypatch, tmp_path):
    monkeypatch.delenv("MANYLATENTS_DATA", raising=False)
    # A repo root is identified by a pyproject.toml sitting next to the package.
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.setattr(dp, "_PKG_PARENT", tmp_path)
    assert omics_data_root() == tmp_path / "data"


def test_installed_wheel_falls_back_to_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("MANYLATENTS_DATA", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    # No pyproject.toml next to the package == an installed wheel in site-packages.
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    monkeypatch.setattr(dp, "_PKG_PARENT", site_packages)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = omics_data_root()
    # Must NOT point inside the install location, and must be user-writable cache.
    assert site_packages not in root.parents
    assert root == tmp_path / "home" / ".cache" / "manylatents" / "data"


def test_xdg_cache_home_respected(monkeypatch, tmp_path):
    monkeypatch.delenv("MANYLATENTS_DATA", raising=False)
    monkeypatch.setattr(dp, "_PKG_PARENT", tmp_path / "site-packages")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert omics_data_root() == tmp_path / "xdg" / "manylatents" / "data"


def test_resolver_registered_and_interpolates():
    # Importing the plugin (as core's entry-point loader does) registers it.
    import manylatents.omics_plugin  # noqa: F401

    assert OmegaConf.has_resolver("omics_data")
    cfg = OmegaConf.create({"p": "${omics_data:}/single_cell/pbmc_3k.h5ad"})
    assert cfg.p == str(omics_data_root() / "single_cell" / "pbmc_3k.h5ad")
