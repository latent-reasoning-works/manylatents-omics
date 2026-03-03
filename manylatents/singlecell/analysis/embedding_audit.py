"""End-to-end embedding fidelity audit: embed -> cluster -> DE -> compare."""
import logging

import numpy as np
import scanpy as sc

from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis
from manylatents.singlecell.analysis.differential_expression import DifferentialExpression

logger = logging.getLogger(__name__)


class EmbeddingAudit:
    """Audit embedding fidelity by comparing DE results across parameter settings."""

    def __init__(self, setting_a: dict, setting_b: dict,
                 leiden_resolution: float = 0.5,
                 de_method: str = "wilcoxon",
                 de_p_threshold: float = 0.05,
                 de_lfc_threshold: float = 1.0):
        self.setting_a = setting_a
        self.setting_b = setting_b
        self.leiden_resolution = leiden_resolution
        self._de_kwargs = dict(
            method=de_method, p_threshold=de_p_threshold, lfc_threshold=de_lfc_threshold
        )
        self.csa = ComplementSetAnalysis()

    def _embed_and_cluster(self, adata: sc.AnnData, setting: dict, suffix: str):
        """Build kNN graph, UMAP embed, Leiden cluster."""
        use_rep = "X_pca" if "X_pca" in adata.obsm else "X"
        sc.pp.neighbors(adata, n_neighbors=setting["n_neighbors"], use_rep=use_rep)
        sc.tl.umap(adata, min_dist=setting.get("min_dist", 0.1))
        umap_key = f"X_umap_{suffix}"
        adata.obsm[umap_key] = adata.obsm["X_umap"].copy()
        leiden_key = f"leiden_{suffix}"
        sc.tl.leiden(adata, resolution=self.leiden_resolution, key_added=leiden_key, flavor="leidenalg")
        n_clusters = adata.obs[leiden_key].nunique()
        logger.info(f"Setting {suffix}: {n_clusters} clusters (k={setting['n_neighbors']})")
        return leiden_key, n_clusters

    def run(self, adata: sc.AnnData) -> dict:
        """Full pipeline: embed -> cluster -> DE -> compare."""
        de_a = DifferentialExpression(**self._de_kwargs)
        de_b = DifferentialExpression(**self._de_kwargs)

        leiden_a, n_clusters_a = self._embed_and_cluster(adata, self.setting_a, "setting_a")
        df_a = de_a.run(adata, groupby=leiden_a, key_added="de_setting_a")
        genes_a = de_a.get_significant_genes()

        leiden_b, n_clusters_b = self._embed_and_cluster(adata, self.setting_b, "setting_b")
        df_b = de_b.run(adata, groupby=leiden_b, key_added="de_setting_b")
        genes_b = de_b.get_significant_genes()

        results = self.csa.compare(genes_a, genes_b, df_a=df_a, df_b=df_b)
        results["setting_a_clusters"] = n_clusters_a
        results["setting_b_clusters"] = n_clusters_b
        results["setting_a"] = self.setting_a
        results["setting_b"] = self.setting_b
        return results

    def plot_comparison(self, adata: sc.AnnData, results: dict, save_path: str = None):
        """Side-by-side embeddings colored by clusters, with complement set summary."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for ax, suffix, title in [
            (axes[0], "setting_a", f"Setting A (k={self.setting_a['n_neighbors']})"),
            (axes[1], "setting_b", f"Setting B (k={self.setting_b['n_neighbors']})"),
        ]:
            coords = adata.obsm[f"X_umap_{suffix}"]
            clusters = adata.obs[f"leiden_{suffix}"].astype(int)
            n_clusters = clusters.nunique()
            cmap = plt.colormaps.get_cmap("tab20").resampled(n_clusters)
            ax.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap=cmap,
                       s=3, alpha=0.7, rasterized=True)
            ax.set_title(f"{title}\n{n_clusters} clusters")
            ax.set_aspect("equal")

        ax = axes[2]
        categories = ["Robust\n(A∩B)", "Artifacts\n(A\\B)", "Missed\n(B\\A)"]
        values = [results["n_robust"], results["n_artifacts"], results["n_missed"]]
        colors = ["#2ecc71", "#e74c3c", "#3498db"]
        bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("DE genes")
        ax.set_title(f"Jaccard = {results['jaccard']:.3f}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
