"""Single-cell analysis tools: DE, complement sets, embedding audits."""
from manylatents.singlecell.analysis.complement_set import ComplementSetAnalysis
from manylatents.singlecell.analysis.differential_expression import DifferentialExpression
from manylatents.singlecell.analysis.embedding_audit import EmbeddingAudit

__all__ = ["ComplementSetAnalysis", "DifferentialExpression", "EmbeddingAudit"]
