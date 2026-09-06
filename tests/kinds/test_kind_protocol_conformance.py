"""LabeledArray satisfies the ``Kind`` protocol (provenance + require + tagged).

THE CONSUMER IS NAMED BY ROLE, NOT BY PROJECT, and that is deliberate rather than
vague. One layer up from this package sits an op registry that types against a
structural ``Kind`` protocol:

    provenance: tuple[str, ...]
    def require(self, *dims, coords=()) -> Kind: ...
    def tagged(self, op_name: str) -> Kind: ...

Any concrete kind is substitutable into that registry iff it offers all three.
The consumer this was first written against is retired, and its successor is
proprietary — so writing either name here would leave a docstring that is wrong,
or one that points at something a reader cannot open. The protocol is the durable
fact; whose registry consumes it is not.

The assertions are STRUCTURAL and import nothing from that layer, which is the
same point said in code: this package is the body, the layer above imports it,
never the reverse. So this repository's CI catches protocol drift on its own,
with no cross-repo dependency to make the arrow point both ways.
"""
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from manylatents.kinds import LabeledArray


def _toy() -> LabeledArray:
    return LabeledArray(
        xr.DataArray(
            np.ones((3, 2)),
            dims=["cell", "gene"],
            coords={"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
    )


class TestKindProtocolConformance:
    def test_offers_all_three_protocol_members(self):
        la = _toy()
        assert isinstance(la.provenance, tuple)
        assert callable(la.require)
        assert callable(la.tagged)

    def test_default_provenance_is_empty(self):
        assert _toy().provenance == ()

    def test_tagged_appends_immutably(self):
        la = _toy()
        out = la.tagged("mean_over_time")
        assert out.provenance == ("mean_over_time",)
        assert la.provenance == ()  # original untouched — tagged returns a copy
        assert isinstance(out, LabeledArray)

    def test_tagged_accumulates(self):
        chained = _toy().tagged("a").tagged("b")
        assert chained.provenance == ("a", "b")

    def test_require_still_returns_self_for_chaining(self):
        la = _toy()
        assert la.require("cell", "gene") is la

    def test_provenance_survives_zarr_round_trip(self):
        tagged = _toy().tagged("mean_over_time").tagged("velocity")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "k.zarr")
            tagged.serialize(path)
            loaded = LabeledArray.load(path)
            assert loaded.provenance == ("mean_over_time", "velocity")
            # provenance is not left dangling as a domain attr
            assert "provenance" not in loaded.da.attrs

    def test_empty_provenance_adds_no_attr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "k.zarr")
            _toy().serialize(path)  # provenance == ()
            loaded = LabeledArray.load(path)
            assert loaded.provenance == ()
            assert "provenance" not in loaded.da.attrs
