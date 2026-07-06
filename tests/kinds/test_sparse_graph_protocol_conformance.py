"""SparseGraph satisfies geomancy's Kind protocol (provenance + require + tagged).

Asserts the surface + behavior *structurally*, with no ``geomancy`` import, so
this repo's CI catches drift on its own (keeps the dependency direction clean).
Mirrors ``test_kind_protocol_conformance`` for the LabeledArray kind.
"""
import tempfile
from pathlib import Path

import numpy as np

from manylatents.kinds import SparseGraph


def _toy() -> SparseGraph:
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    node_ids = np.array([0, 1, 2])
    return SparseGraph(edges, node_ids)


class TestKindProtocolConformance:
    def test_offers_all_three_protocol_members(self):
        g = _toy()
        assert isinstance(g.provenance, tuple)
        assert callable(g.require) and callable(g.tagged)

    def test_default_provenance_is_empty(self):
        assert _toy().provenance == ()

    def test_provenance_normalized_to_tuple(self):
        g = SparseGraph(_toy().edges, _toy().node_ids, provenance=["a", "b"])
        assert g.provenance == ("a", "b")

    def test_tagged_appends_immutably(self):
        g = _toy()
        out = g.tagged("prune_edges")
        assert out.provenance == ("prune_edges",)
        assert g.provenance == ()  # original untouched
        assert isinstance(out, SparseGraph)

    def test_tagged_accumulates(self):
        assert _toy().tagged("a").tagged("b").provenance == ("a", "b")

    def test_require_still_returns_self_for_chaining(self):
        g = _toy()
        assert g.require("edges", "node_ids") is g

    def test_provenance_survives_npz_round_trip(self):
        tagged = _toy().tagged("prune_edges").tagged("relabel")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "g.npz")
            tagged.serialize(path)
            loaded = SparseGraph.load(path)
            assert loaded.provenance == ("prune_edges", "relabel")

    def test_empty_provenance_round_trips_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "g.npz")
            _toy().serialize(path)
            loaded = SparseGraph.load(path)
            assert loaded.provenance == ()
