"""LARRY lineage-tracing dataset: config, offline build from cached source files, datamodule.

The real source files are about 2 GB; the build is exercised on miniature files written in
the published layout. Only ``test_source_urls_are_reachable`` touches the network.
"""

import gzip
import urllib.error
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf
from scipy import sparse
from scipy.io import mmwrite

import manylatents.omics_plugin  # noqa: F401  Register the data-path resolver.
from manylatents.singlecell.data import larry

CONFIG_DIR = Path(__file__).resolve().parents[2] / "manylatents/singlecell/configs/data"
SMALL = {"n_top_genes": 30, "n_comps": 5}


def write_source_files(raw_dir: Path, n_cells=150, n_genes=60, n_clones=20, seed=0):
    """Write the four stateFate_inVitro_* files with the published shapes and columns."""
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts = sparse.random(n_cells, n_genes, density=0.5, random_state=seed, format="coo") * 20
    with gzip.open(raw_dir / larry.SOURCE_FILES["counts"], "wb") as f:
        mmwrite(f, counts)
    genes = [f"Gene{i}" for i in range(n_genes - 1)] + ["Gene0"]  # one duplicate name
    with gzip.open(raw_dir / larry.SOURCE_FILES["genes"], "wt") as f:
        f.write("\n".join(genes) + "\n")
    clone = rng.integers(0, n_clones, size=n_cells)
    clone[:5] = -1  # five cells carry no barcode
    rows = np.flatnonzero(clone >= 0)
    clones = sparse.coo_matrix((np.ones(rows.size), (rows, clone[rows])), shape=(n_cells, n_clones))
    with gzip.open(raw_dir / larry.SOURCE_FILES["clones"], "wb") as f:
        mmwrite(f, clones)
    days = rng.choice([2, 4, 6], size=n_cells)
    days[5:8] = 3  # not one of the three sampled days kept
    types = rng.choice(["Neutrophil", "Monocyte", "Undifferentiated", "Baso", "Meg"], size=n_cells)
    meta = pd.DataFrame({
        "Library": "d2_1", "Cell barcode": [f"BC{i}" for i in range(n_cells)],
        "Time point": days, "Starting population": "Lin-Kit+Sca1-",
        "Cell type annotation": types, "Well": 0,
        "SPRING-x": rng.normal(size=n_cells), "SPRING-y": rng.normal(size=n_cells),
    })
    with gzip.open(raw_dir / larry.SOURCE_FILES["metadata"], "wt") as f:
        meta.to_csv(f, sep="\t", index=False)
    return clone, days, types


def forbid_network(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Cached source files must not trigger a download")
    monkeypatch.setattr(larry, "_download", no_network)


def test_config_targets_the_larry_datamodule(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYLATENTS_DATA", str(tmp_path))
    assert CONFIG_DIR.is_dir()
    path = CONFIG_DIR / "larry.yaml"
    config = OmegaConf.load(path)
    assert get_class(config._target_) is larry.LarryDataModule
    assert Path(config.adata_path) == tmp_path / "single_cell" / "larry.h5ad"
    assert Path(config.adata_path) == larry.default_path()
    assert config.label_key == "fate"
    text = path.read_text()
    assert larry.SOURCE_BASE_URL in text
    assert larry.DOI in text


def test_source_urls_are_public_https():
    assert larry.SOURCE_BASE_URL.startswith("https://")
    assert set(larry.SOURCE_FILES) == {"counts", "genes", "clones", "metadata"}
    for url in larry.source_urls().values():
        assert url.startswith(larry.SOURCE_BASE_URL)
        assert url.rsplit("/", 1)[1].startswith("stateFate_inVitro_")


def test_build_anndata_maps_source_columns(tmp_path):
    clone, days, types = write_source_files(tmp_path)
    adata = larry.build_anndata(tmp_path)
    keep = (clone >= 0) & np.isin(days, [2, 4, 6])
    assert adata.n_obs == keep.sum()
    assert adata.obs_names.tolist() == [f"c{i}" for i in np.flatnonzero(keep)]
    assert adata.obs["clone"].to_numpy().tolist() == clone[keep].tolist()
    assert adata.obs["Time point"].to_numpy().tolist() == days[keep].tolist()
    want_fate = [larry.FATE_MAP.get(t, "other") for t in types[keep]]
    assert adata.obs["fate"].astype(str).tolist() == want_fate
    assert {"Cell type annotation", "Library", "SPRING-x"} <= set(adata.obs.columns)
    assert adata.var_names.is_unique
    assert adata.X.dtype == np.float32


def test_fetch_builds_once_from_cached_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYLATENTS_DATA", str(tmp_path))
    write_source_files(larry.default_raw_dir())
    forbid_network(monkeypatch)
    path = larry.fetch_larry(**SMALL)
    assert path == larry.default_path()
    result = ad.read_h5ad(path)
    assert result.obsm["X_pca"].shape == (result.n_obs, SMALL["n_comps"])
    assert result.obsm["X_pca"].dtype == np.float32
    assert result.n_vars <= SMALL["n_top_genes"]
    assert "normalized" in result.layers
    assert {"Time point", "clone", "fate"} <= set(result.obs.columns)
    assert result.uns["source_url"] == larry.SOURCE_BASE_URL
    assert result.uns["doi"] == larry.DOI

    stamp = path.stat().st_mtime_ns
    monkeypatch.setattr(larry, "build_anndata", lambda *a, **k: pytest.fail("rebuilt a cached file"))
    assert larry.fetch_larry(**SMALL) == path
    assert path.stat().st_mtime_ns == stamp


def test_fetch_downloads_missing_sources(tmp_path, monkeypatch):
    source = tmp_path / "published"
    write_source_files(source)
    fetched = []

    def fake_download(url, dest):
        fetched.append(url)
        dest.write_bytes((source / url.rsplit("/", 1)[1]).read_bytes())

    monkeypatch.setattr(larry, "_download", fake_download)
    out = tmp_path / "cache" / "larry.h5ad"
    larry.fetch_larry(out, **SMALL)
    assert sorted(fetched) == sorted(larry.source_urls().values())
    assert out.is_file()


def test_datamodule_builds_on_first_use(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYLATENTS_DATA", str(tmp_path))
    write_source_files(larry.default_raw_dir())
    forbid_network(monkeypatch)
    monkeypatch.setattr(larry, "PREPROCESS_DEFAULTS", SMALL)
    dm = larry.LarryDataModule(adata_path=str(larry.default_path()), label_key="fate")
    dm.setup()
    assert len(dm.train_dataset) == ad.read_h5ad(larry.default_path()).n_obs
    assert dm.train_dataset.get_obs("clone") is not None


@pytest.mark.network
def test_source_urls_are_reachable():
    for url in larry.source_urls().values():
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8.7.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                assert r.status == 200
        except (urllib.error.URLError, TimeoutError) as e:
            pytest.skip(f"source host unreachable: {e}")
