"""Tests for DifferentialExpression wrapper."""
import numpy as np
import pytest

sc = pytest.importorskip("scanpy")


def _make_test_adata(n_cells=200, n_genes=50, n_clusters=3, seed=42):
    """Create a minimal AnnData with cluster assignments for DE testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_cells, n_genes).astype(np.float32)
    labels = np.array([i % n_clusters for i in range(n_cells)])
    # Boost genes 0-4 in cluster 0
    cluster_0_mask = labels == 0
    X[cluster_0_mask, :5] += 5.0

    import anndata
    adata = anndata.AnnData(
        X=X,
        obs={"cluster": labels.astype(str)},
    )
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    return adata


class TestDifferentialExpression:
    def test_run_returns_dataframe(self):
        from manylatents.singlecell.analysis.differential_expression import DifferentialExpression
        adata = _make_test_adata()
        de = DifferentialExpression(method="wilcoxon")
        df = de.run(adata, groupby="cluster")
        assert "gene" in df.columns
        assert "cluster" in df.columns
        assert "pval_adj" in df.columns
        assert "logfoldchange" in df.columns
        assert len(df) > 0

    def test_get_significant_genes(self):
        from manylatents.singlecell.analysis.differential_expression import DifferentialExpression
        adata = _make_test_adata()
        de = DifferentialExpression(method="wilcoxon", p_threshold=0.05, lfc_threshold=0.5)
        de.run(adata, groupby="cluster")
        genes = de.get_significant_genes()
        assert isinstance(genes, set)
        assert len(genes) > 0

    def test_boosted_genes_detected(self):
        from manylatents.singlecell.analysis.differential_expression import DifferentialExpression
        adata = _make_test_adata(n_cells=300)
        de = DifferentialExpression(method="wilcoxon", p_threshold=0.05, lfc_threshold=1.0)
        de.run(adata, groupby="cluster")
        genes = de.get_significant_genes()
        boosted = {f"gene_{i}" for i in range(5)}
        assert boosted & genes, f"Expected some of {boosted} in significant genes {genes}"
