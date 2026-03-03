"""Tests for ComplementSetAnalysis."""
import pytest


class TestComplementSetAnalysis:
    def test_basic_set_operations(self):
        from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis
        csa = ComplementSetAnalysis()
        genes_a = {"A", "B", "C", "D"}
        genes_b = {"C", "D", "E", "F"}
        result = csa.compare(genes_a, genes_b)
        assert result["robust"] == {"C", "D"}
        assert result["artifacts"] == {"A", "B"}
        assert result["missed"] == {"E", "F"}
        assert result["n_robust"] == 2
        assert result["n_artifacts"] == 2
        assert result["n_missed"] == 2
        assert abs(result["jaccard"] - 2 / 6) < 1e-9

    def test_empty_complement(self):
        from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis
        csa = ComplementSetAnalysis()
        genes = {"A", "B", "C"}
        result = csa.compare(genes, genes)
        assert result["n_artifacts"] == 0
        assert result["n_missed"] == 0
        assert result["jaccard"] == 1.0

    def test_disjoint_sets(self):
        from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis
        csa = ComplementSetAnalysis()
        result = csa.compare({"A", "B"}, {"C", "D"})
        assert result["n_robust"] == 0
        assert result["jaccard"] == 0.0

    def test_empty_inputs(self):
        from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis
        csa = ComplementSetAnalysis()
        result = csa.compare(set(), set())
        assert result["n_robust"] == 0
        assert result["jaccard"] == 0.0
