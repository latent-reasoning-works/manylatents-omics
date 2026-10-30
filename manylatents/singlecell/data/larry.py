"""LARRY lineage-traced haematopoiesis (Weinreb et al. 2020), fetched on first use.

Weinreb, C., Rodriguez-Fraticelli, A., Camargo, F. D. & Klein, A. M. Lineage tracing on
transcriptional landscapes links state to fate during differentiation. Science 367,
eaaw3381 (2020). doi:10.1126/science.aaw3381

Source: the in-vitro experiment files published on the paper website,
``https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_*``:

=========================================  ==============================================
``normed_counts.mtx.gz``                   cells x genes, total-count normalised (not integer)
``gene_names.txt.gz``                      one gene symbol per line
``clone_matrix.mtx.gz``                    cells x clones, 1 where a cell carries a barcode
``metadata.txt.gz``                        per-cell table (``Time point``, ``Cell type annotation``, ...)
=========================================  ==============================================

How the processed file maps to the source:

1. ``obs`` is the metadata table, in source order, with ``obs_names`` ``c<row>`` (source row).
2. ``obs["clone"]`` is the column index of each cell's barcode in the clone matrix (the
   first one if a cell carries several); ``-1`` for cells without one.
3. ``obs["fate"]`` maps ``Cell type annotation``: Neutrophil -> neutrophil, Monocyte ->
   monocyte, Undifferentiated -> undiff, every other type -> other.
4. Cells without a clone, or outside time points 2, 4 and 6, are dropped.
5. Genes and cells with total < 3 are dropped; ``layers["normalized"]`` keeps the source
   values; ``X`` is ``log1p`` of them restricted to 2,000 highly variable genes (scanpy
   ``seurat`` flavour); ``obsm["X_pca"]`` is a 50-component ARPACK PCA as float32.
   The source is already normalised, so no ``normalize_total`` step is applied.

The four source files (about 2.1 GB, mostly the count matrix) and the processed file are
written below :func:`manylatents._data_paths.omics_data_root` and never re-downloaded.
"""

from __future__ import annotations

import gzip
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from manylatents._data_paths import omics_data_root

from .anndata import AnnDataModule

logger = logging.getLogger(__name__)

DOI = "10.1126/science.aaw3381"
SOURCE_BASE_URL = "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/"
SOURCE_FILES = {
    "counts": "stateFate_inVitro_normed_counts.mtx.gz",
    "genes": "stateFate_inVitro_gene_names.txt.gz",
    "clones": "stateFate_inVitro_clone_matrix.mtx.gz",
    "metadata": "stateFate_inVitro_metadata.txt.gz",
}
# The metadata file spells the progenitor state "Undifferentiated" (not "undiff").
FATE_MAP = {"Neutrophil": "neutrophil", "Monocyte": "monocyte", "Undifferentiated": "undiff"}
TIME_POINTS = (2, 4, 6)
PREPROCESS_DEFAULTS = {"n_top_genes": 2000, "n_comps": 50}


def source_urls() -> dict[str, str]:
    """Public URL of each source file, keyed like :data:`SOURCE_FILES`."""
    return {k: SOURCE_BASE_URL + name for k, name in SOURCE_FILES.items()}


def default_path() -> Path:
    """Where the processed file lives: ``<omics data root>/single_cell/larry.h5ad``."""
    return omics_data_root() / "single_cell" / "larry.h5ad"


def default_raw_dir() -> Path:
    """Where the source files are cached: ``<omics data root>/single_cell/larry_source``."""
    return default_path().parent / "larry_source"


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` through a ``.part`` file, so no partial file is cached."""
    partial = dest.with_name(dest.name + ".part")
    logger.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.7.1"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r, open(partial, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def download_sources(raw_dir: Path) -> dict[str, Path]:
    """Download any source file missing from ``raw_dir``; return their paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, url in source_urls().items():
        dest = raw_dir / SOURCE_FILES[key]
        if not dest.exists():
            _download(url, dest)
        paths[key] = dest
    return paths


def build_anndata(raw_dir: Path):
    """Assemble the unprocessed AnnData (steps 1-4 of the module docstring)."""
    import anndata as ad
    import scipy.io

    raw_dir = Path(raw_dir)
    with gzip.open(raw_dir / SOURCE_FILES["counts"]) as f:
        X = scipy.io.mmread(f).tocsr().astype(np.float32)
    with gzip.open(raw_dir / SOURCE_FILES["genes"], "rt") as f:
        genes = [g.strip() for g in f]
    md = pd.read_csv(raw_dir / SOURCE_FILES["metadata"], sep="\t")
    with gzip.open(raw_dir / SOURCE_FILES["clones"]) as f:
        clones = scipy.io.mmread(f).tocsr()
    clone_id = np.where(clones.getnnz(axis=1) > 0, np.asarray(clones.argmax(axis=1)).ravel(), -1)

    adata = ad.AnnData(X=X, obs=md.reset_index(drop=True))
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    adata.var_names = genes
    adata.var_names_make_unique()
    adata.obs["clone"] = clone_id
    adata.obs["fate"] = [FATE_MAP.get(t, "other") for t in adata.obs["Cell type annotation"].astype(str)]
    keep = (adata.obs["clone"] >= 0).values & adata.obs["Time point"].isin(TIME_POINTS).values
    return adata[keep].copy()


def preprocess(adata, n_top_genes: int = 2000, n_comps: int = 50):
    """Step 5 of the module docstring. Returns a new AnnData; ``adata`` is modified."""
    import scanpy as sc

    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=3)
    adata.layers["normalized"] = adata.X.copy()
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
    adata.obsm["X_pca"] = adata.obsm["X_pca"].astype(np.float32)
    return adata


def _object_strings(adata) -> None:
    """Store string and categorical columns as plain objects, which every anndata writes."""
    for df in (adata.obs, adata.var):
        df.index = pd.Index(df.index.astype(object).tolist(), dtype=object)
        for c in df.columns:
            if isinstance(df[c].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(df[c]):
                df[c] = np.asarray(df[c].astype(str).tolist(), dtype=object)


def fetch_larry(
    path: Path | None = None,
    *,
    raw_dir: Path | None = None,
    n_top_genes: int | None = None,
    n_comps: int | None = None,
) -> Path:
    """Return the processed LARRY ``.h5ad``, downloading and building it if absent.

    Args:
        path: Processed file location; defaults to :func:`default_path`.
        raw_dir: Source-file cache; defaults to ``larry_source/`` next to ``path``.
        n_top_genes, n_comps: Override the preprocessing defaults (2,000 and 50).
    """
    path = Path(path) if path is not None else default_path()
    if path.exists():
        return path
    raw_dir = Path(raw_dir) if raw_dir is not None else path.parent / "larry_source"
    download_sources(raw_dir)
    params = {**PREPROCESS_DEFAULTS}
    if n_top_genes is not None:
        params["n_top_genes"] = n_top_genes
    if n_comps is not None:
        params["n_comps"] = n_comps
    adata = preprocess(build_anndata(raw_dir), **params)
    adata.uns["source_url"] = SOURCE_BASE_URL
    adata.uns["doi"] = DOI
    _object_strings(adata)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".part" + path.suffix)
    try:
        adata.write_h5ad(partial, compression="gzip")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    logger.info("Saved LARRY: %s (%d cells x %d genes)", path, adata.n_obs, adata.n_vars)
    return path


def load_larry(path: Path | None = None):
    """Read the processed LARRY AnnData, fetching it on first use."""
    import anndata as ad

    return ad.read_h5ad(fetch_larry(path))


class LarryDataModule(AnnDataModule):
    """:class:`AnnDataModule` that fetches and builds the LARRY file on first use."""

    def _ensure(self) -> None:
        if not Path(self.adata_path).exists():
            fetch_larry(Path(self.adata_path))

    def prepare_data(self) -> None:
        self._ensure()

    def setup(self, stage: str | None = None) -> None:
        self._ensure()
        super().setup(stage)
