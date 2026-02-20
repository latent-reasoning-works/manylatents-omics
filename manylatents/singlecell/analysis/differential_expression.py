"""Differential expression wrapper around scanpy."""
import pandas as pd
import scanpy as sc


class DifferentialExpression:
    """Run DE on an AnnData object given cluster assignments."""

    def __init__(self, method: str = "wilcoxon", p_threshold: float = 0.05,
                 lfc_threshold: float = 1.0, n_genes: int = 200):
        self.method = method
        self.p_threshold = p_threshold
        self.lfc_threshold = lfc_threshold
        self.n_genes = n_genes
        self._df = None

    def run(self, adata: sc.AnnData, groupby: str, key_added: str = "de") -> pd.DataFrame:
        """Run rank_genes_groups, return tidy DataFrame."""
        self._key = key_added
        sc.tl.rank_genes_groups(adata, groupby=groupby, method=self.method,
                                key_added=key_added, n_genes=self.n_genes)
        result = adata.uns[key_added]
        groups = result["names"].dtype.names
        rows = []
        for group in groups:
            for i in range(len(result["names"][group])):
                rows.append({
                    "gene": result["names"][group][i],
                    "cluster": group,
                    "pval_adj": result["pvals_adj"][group][i],
                    "logfoldchange": result["logfoldchanges"][group][i],
                    "score": result["scores"][group][i],
                })
        self._df = pd.DataFrame(rows)
        return self._df

    def get_significant_genes(self, adata: sc.AnnData, key: str = None) -> set:
        """Extract set of significant gene names."""
        if self._df is None:
            raise RuntimeError("Call run() before get_significant_genes()")
        sig = self._df[
            (self._df["pval_adj"] < self.p_threshold)
            & (self._df["logfoldchange"].abs() > self.lfc_threshold)
        ]
        return set(sig["gene"].unique())
