"""Compare DE gene lists across embedding parameter settings."""
from __future__ import annotations

import pandas as pd


class ComplementSetAnalysis:
    """Compare DE results from two parameter settings."""

    def compare(self, genes_a: set, genes_b: set,
                df_a: pd.DataFrame | None = None, df_b: pd.DataFrame | None = None) -> dict:
        """Compute complement sets with optional per-gene stats."""
        robust = genes_a & genes_b
        artifacts = genes_a - genes_b
        missed = genes_b - genes_a
        union = genes_a | genes_b
        jaccard = len(robust) / len(union) if union else 0.0

        artifact_df = None
        if df_a is not None and len(artifacts) > 0:
            artifact_df = df_a[df_a["gene"].isin(artifacts)].copy()

        return {
            "robust": robust,
            "artifacts": artifacts,
            "missed": missed,
            "n_robust": len(robust),
            "n_artifacts": len(artifacts),
            "n_missed": len(missed),
            "jaccard": jaccard,
            "artifact_genes_df": artifact_df,
        }
