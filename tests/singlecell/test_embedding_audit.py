"""Tests for EmbeddingAudit end-to-end pipeline."""
import numpy as np
import pytest

sc = pytest.importorskip("scanpy")
pytest.importorskip("leidenalg")


def _make_trajectory_adata(n_cells=300, n_genes=100, seed=42):
    """Create synthetic trajectory data."""
    rng = np.random.RandomState(seed)
    pseudotime = np.linspace(0, 1, n_cells)
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for g in range(n_genes):
        freq = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        X[:, g] = np.sin(freq * pseudotime * 2 * np.pi + phase) + rng.randn(n_cells) * 0.3

    import anndata
    adata = anndata.AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    sc.tl.pca(adata, n_comps=min(50, n_genes - 1))
    return adata


class TestEmbeddingAudit:
    def test_run_returns_complement_results(self):
        from manylatents.singlecell.analysis.embedding_audit import EmbeddingAudit
        adata = _make_trajectory_adata()
        audit = EmbeddingAudit(
            setting_a={"n_neighbors": 5, "min_dist": 0.01},
            setting_b={"n_neighbors": 30, "min_dist": 0.5},
            leiden_resolution=0.5,
        )
        results = audit.run(adata)
        assert "n_robust" in results
        assert "n_artifacts" in results
        assert "n_missed" in results
        assert "jaccard" in results
        assert "setting_a_clusters" in results
        assert "setting_b_clusters" in results

    def test_run_preserves_embeddings(self):
        from manylatents.singlecell.analysis.embedding_audit import EmbeddingAudit
        adata = _make_trajectory_adata()
        audit = EmbeddingAudit(
            setting_a={"n_neighbors": 5, "min_dist": 0.01},
            setting_b={"n_neighbors": 30, "min_dist": 0.5},
        )
        audit.run(adata)
        assert "X_umap_setting_a" in adata.obsm
        assert "X_umap_setting_b" in adata.obsm
