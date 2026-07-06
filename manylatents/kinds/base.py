"""
Base class for typed internal data representations (kinds).

Each kind carries its own structural semantics (dims, labels, coords).
This ensures ops can read and validate structure instead of guessing.
"""

from abc import ABC, abstractmethod


class Kind(ABC):
    """Base class for all data kinds.

    Each kind owns its own structural semantics *and* its own persistence:
    subclasses must implement ``validate``, ``serialize``, and ``load`` for the
    storage format appropriate to that kind. The base declares only the contract.
    """

    @abstractmethod
    def validate(self) -> "Kind":
        """Validate the kind's structure. Raise on failure."""
        ...

    @abstractmethod
    def serialize(self, path: str) -> None:
        """Write the kind to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Kind":
        """Load the kind from disk, validating on read."""
        ...

    @abstractmethod
    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "Kind":
        """Assert named dims/coords are present; raise cleanly if not. Returns self."""
        ...

    @abstractmethod
    def tagged(self, op_name: str) -> "Kind":
        """Return a copy with ``op_name`` appended to the provenance trail."""
        ...
