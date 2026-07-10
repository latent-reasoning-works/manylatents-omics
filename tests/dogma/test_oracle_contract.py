"""Smoke test for the SignalOracle contract + registry (no models required).

Exercises the row #29 deliverable end to end with a model-free dummy oracle:
structural conformance, registry roundtrip, record emission, and the
``layers=`` filter. Runnable via plain ``python3`` or ``pytest`` -- imports
nothing heavier than ``manylatents.dogma.oracles`` (typing only).

The ``Variant`` / ``SignalRecord`` stand-ins below mirror the #26 schema shape
so the test can run before that module lands; a conforming oracle would emit
the real types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import pytest

from manylatents.dogma.oracles import (
    SIGNAL_LAYERS,
    SignalOracle,
    get_oracle,
    list_oracles,
    register_oracle,
)


# --- Stand-ins for the #26 signal-schema types (test-local only) -----------


class Variant(NamedTuple):
    chrom: str
    pos: int
    ref: str
    alt: str
    id: str


@dataclass
class StubSignalRecord:
    variant: Variant
    track_id: str
    layer: str
    cell_type: str
    delta: float
    effect_pctl: float
    activity_pctl: float


# --- A model-free oracle implementing the Protocol -------------------------


@register_oracle("dummy")
class DummyOracle:
    """Emits deterministic records; no model, no I/O."""

    oracle_id = "dummy"
    layers = ("accessibility", "tf")
    cell_types = ("K562",)
    track_ids = ("dummy:accessibility:K562", "dummy:tf:K562")

    def score_variant(
        self,
        variant: Variant,
        *,
        layers: Optional[Sequence[str]] = None,
    ) -> Sequence[StubSignalRecord]:
        wanted = set(layers) if layers is not None else set(self.layers)
        records = []
        for layer, track in zip(self.layers, self.track_ids):
            if layer not in wanted:
                continue
            records.append(
                StubSignalRecord(
                    variant=variant,
                    track_id=track,
                    layer=layer,
                    cell_type=self.cell_types[0],
                    delta=0.5,
                    effect_pctl=90.0,
                    activity_pctl=80.0,
                )
            )
        return records


_VARIANT = Variant(chrom="chr1", pos=1000, ref="A", alt="G", id="rs_test")


def test_dummy_conforms_to_protocol():
    assert isinstance(DummyOracle(), SignalOracle)


def test_advertised_layers_are_in_vocabulary():
    for layer in DummyOracle().layers:
        assert layer in SIGNAL_LAYERS


def test_registry_roundtrip():
    assert "dummy" in list_oracles()
    assert get_oracle("dummy") is DummyOracle
    assert get_oracle("DUMMY") is DummyOracle  # case-insensitive


def test_get_unknown_oracle_raises():
    with pytest.raises(KeyError):
        get_oracle("does_not_exist")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):
        register_oracle("dummy")(type("Other", (), {}))


def test_score_variant_emits_one_record_per_track():
    records = DummyOracle().score_variant(_VARIANT)
    assert len(records) == 2
    assert {r.track_id for r in records} == set(DummyOracle.track_ids)
    for r in records:
        assert r.variant == _VARIANT
        assert r.layer in SIGNAL_LAYERS


def test_layers_filter_restricts_output():
    records = DummyOracle().score_variant(_VARIANT, layers=["tf"])
    assert len(records) == 1
    assert records[0].layer == "tf"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
