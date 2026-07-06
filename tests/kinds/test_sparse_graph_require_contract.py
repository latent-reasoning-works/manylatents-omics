"""Op contracts: requiring named components via ``require`` on SparseGraph.

Required components are enforced structurally by ``require`` — the contract an op
declares for the kind it consumes. Mirrors ``test_require_contract`` for the
LabeledArray kind; a SparseGraph carries the ``edges``/``node_ids`` components and
no coords.
"""

import numpy as np
import pytest

from manylatents.kinds import SparseGraph


def toy_graph() -> SparseGraph:
    return SparseGraph(np.array([[0, 1], [1, 2], [2, 0]]), np.array([0, 1, 2]))


class TestRequireContract:
    """Ops declare and enforce the components they consume via ``kind.require``."""

    def test_require_passes_with_present_components(self):
        g = toy_graph()
        assert g.require("edges", "node_ids") is g  # returns self for chaining

    def test_require_passes_with_subset(self):
        g = toy_graph()
        assert g.require("edges") is g

    def test_require_rejects_missing_component(self):
        with pytest.raises(ValueError, match="requires dims"):
            toy_graph().require("cell", "gene")

    def test_require_rejects_unknown_component(self):
        with pytest.raises(ValueError, match="requires dims"):
            toy_graph().require("edges", "weights")

    def test_require_rejects_any_coord(self):
        # a bare edge list + node ids carries no coords, so any coord fails.
        with pytest.raises(ValueError, match="requires coords"):
            toy_graph().require("edges", "node_ids", coords=("node_labels",))
