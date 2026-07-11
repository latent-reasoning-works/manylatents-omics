"""Signal-oracle contract for the dogma signal manifold (v0).

An *oracle* turns one variant into per-track ``SignalRecord``s and advertises
the axes those records span, so the fusion seam can align tracks across
oracles. This is the seam that lets oracle #2 (ChromBPNet / Borzoi) and later
RNA-processing experts drop in behind a single, tiny interface.

Scope note -- this module deliberately stays minimal:

* ``SignalRecord`` / ``Variant`` are owned by the signal-schema milestone
  (#26) and referenced structurally (``TYPE_CHECKING`` only); this contract
  does not hard-depend on the schema module before it lands.
* ``SignalOracle`` generalizes exactly one method the existing encoders
  already share -- ``AlphaGenomeEncoder.predict()`` (sequence -> regulatory
  tracks) -- plus lightweight introspection modelled on the encoder
  ``modality`` / ``embedding_dim`` property pattern. Everything else on
  ``FoundationEncoder`` (dense embeddings, GPU batching, device juggling) is
  intentionally excluded: an oracle emits *signal*, it is not an embedding
  backbone.

See ``docs/m-v0-oracle-contract.md`` for the design note and next-oracle
shortlist.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Type,
    runtime_checkable,
)

if TYPE_CHECKING:
    # Owned by the #26 signal-schema milestone. Referenced by name only so the
    # oracle contract can land without waiting on the schema module.
    from manylatents.dogma.signal import SignalRecord, Variant


# v0 layer vocabulary each oracle draws ``layers`` from. The #26 ``SignalRecord``
# schema is the canonical owner; now that it has landed, import it rather than
# keep a duplicate mirror in sync. Same-package import (both live in
# manylatents.dogma). NOTE: the core-side geometry (manylatents.metrics) keeps
# its own copy on purpose — core cannot depend on omics, so the canonical vocab
# cannot be centralized here if core must also name it.
from manylatents.dogma.signal import CANONICAL_LAYERS as SIGNAL_LAYERS


@runtime_checkable
class SignalOracle(Protocol):
    """Minimal contract any oracle / tiny-expert implements.

    Two responsibilities:

    1. **Emit** -- ``score_variant(variant)`` returns one ``SignalRecord`` per
       (variant, track). This generalizes ``AlphaGenomeEncoder.predict()``.
    2. **Advertise** -- ``oracle_id`` / ``layers`` / ``cell_types`` /
       ``track_ids`` let the fusion seam and registry plan a consistent
       manifold across oracles *without* running the model.

    Structural (``Protocol``) rather than an ABC on purpose: an oracle may wrap
    an existing ``FoundationEncoder`` (e.g. AlphaGenome), a fresh model
    (ChromBPNet/Borzoi), or a pure-Python expert -- none of which should be
    forced to subclass a shared base. ``@runtime_checkable`` so the registry
    can sanity-check registrations.
    """

    @property
    def oracle_id(self) -> str:
        """Stable identifier. Written into each record's provenance and used as
        the registry key (cf. encoder ``model_name``)."""
        ...

    @property
    def layers(self) -> Sequence[str]:
        """Signal layers this oracle emits, each drawn from ``SIGNAL_LAYERS``."""
        ...

    @property
    def cell_types(self) -> Sequence[str]:
        """Cell types / ontology terms this oracle resolves. May be empty for
        cell-type-agnostic oracles (e.g. a splicing expert)."""
        ...

    @property
    def track_ids(self) -> Sequence[str]:
        """Stable per-track identifiers -- the join key the fusion seam uses to
        align columns across oracles."""
        ...

    def score_variant(
        self,
        variant: "Variant",
        *,
        layers: Optional[Sequence[str]] = None,
    ) -> "Sequence[SignalRecord]":
        """Emit one ``SignalRecord`` per (variant, track).

        Args:
            variant: variant key (chrom, pos, ref, alt, id) -- the #26 type.
            layers: optional subset of ``self.layers`` to restrict output to;
                mirrors ``AlphaGenomeEncoder.predict(output_types=...)``.
                ``None`` (default) emits every available layer.

        Returns:
            One ``SignalRecord`` per emitted track (the #26 schema).
        """
        ...


# ---------------------------------------------------------------------------
# Registry -- lightweight decorator lookup, mirroring metrics/registry.py.
# Oracle implementations live in sibling modules (chrombpnet.py, borzoi.py, ...)
# and register themselves on import.
# ---------------------------------------------------------------------------

_ORACLE_REGISTRY: Dict[str, Type["SignalOracle"]] = {}


def register_oracle(name: str):
    """Class decorator: register an oracle under ``name`` for lookup.

    Example:
        >>> @register_oracle("chrombpnet")
        ... class ChromBPNetOracle:
        ...     ...
    """

    def _decorator(cls: Type["SignalOracle"]) -> Type["SignalOracle"]:
        key = name.lower()
        existing = _ORACLE_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"oracle {name!r} already registered to {existing.__name__!r}"
            )
        _ORACLE_REGISTRY[key] = cls
        return cls

    return _decorator


def get_oracle(name: str) -> Type["SignalOracle"]:
    """Return a registered oracle class by name (case-insensitive)."""
    key = name.lower()
    if key not in _ORACLE_REGISTRY:
        raise KeyError(
            f"oracle {name!r} not found; available: {list_oracles()}"
        )
    return _ORACLE_REGISTRY[key]


def list_oracles() -> List[str]:
    """Return the sorted names of all registered oracles."""
    return sorted(_ORACLE_REGISTRY)


__all__ = [
    "SIGNAL_LAYERS",
    "SignalOracle",
    "register_oracle",
    "get_oracle",
    "list_oracles",
]
