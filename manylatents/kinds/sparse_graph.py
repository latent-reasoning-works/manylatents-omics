"""
SparseGraph: a graph as two plain numpy arrays (edge list + node ids).
"""

import logging
from dataclasses import dataclass

import numpy as np

from .base import Kind

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class SparseGraph(Kind):
    """2 np arrays: edge list and node ids"""

    # named structural components this kind carries; ``require`` checks against these.
    _COMPONENTS = ("edges", "node_ids")

    edges: np.ndarray
    node_ids: np.ndarray
    provenance: tuple[str, ...] = ()

    def __post_init__(self):
        # frozen: bypass the immutability guard to normalize inputs in place.
        object.__setattr__(self, "edges", np.asarray(self.edges))
        object.__setattr__(self, "node_ids", np.asarray(self.node_ids))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        self.validate()

    def validate(self) -> "SparseGraph":
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError(f"edges must be E×2, got shape {self.edges.shape}")
        if self.node_ids.ndim != 1:
            raise ValueError(f"node_ids must be 1-D, got shape {self.node_ids.shape}")
        if not np.issubdtype(self.edges.dtype, np.integer):
            raise ValueError(f"edges must be integer dtype, got {self.edges.dtype}")
        return self

    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "SparseGraph":
        missing_dims = [d for d in dims if d not in self._COMPONENTS]
        if missing_dims:
            raise ValueError(f"requires dims {missing_dims}; got {self._COMPONENTS}")

        # A SparseGraph is a bare edge list + node ids; it carries no coords.
        if coords:
            raise ValueError(f"requires coords {list(coords)}; got ()")
        return self

    def tagged(self, op_name: str) -> "SparseGraph":
        """
        Return a copy with ``op_name`` appended to the provenance trail.
        Part of the geomancy ``Kind`` protocol: geomancy's op registry appends to
        this trail as it runs each op. Immutable: the original is untouched.
        """
        return SparseGraph(self.edges, self.node_ids, self.provenance + (op_name,))

    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".npz"):
            raise ValueError(f"path must end in .npz, got {path!r}")
        return str(path)

    def serialize(self, path: str) -> None:
        logger.info(f"Serializing {type(self).__name__} to {path}")
        # provenance rides alongside as a string array so the op trail survives
        # the round-trip; stored explicitly since .npz keys are arrays, not attrs.
        np.savez_compressed(
            self._normalize(path),
            edges=self.edges,
            node_ids=self.node_ids,
            provenance=np.asarray(self.provenance, dtype=object),
        )

    @classmethod
    def load(cls, path):

        with np.load(cls._normalize(path), allow_pickle=True) as d:
            # older archives predate provenance; default to an empty trail.
            provenance = tuple(d["provenance"]) if "provenance" in d else ()
            # validate called from __post_init__
            return cls(d['edges'], d['node_ids'], provenance)

    @property
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.edges, self.node_ids

    def __repr__(self) -> str:
        return f"SparseGraph(num_nodes={self.node_ids.shape[0]}, num_edges={self.edges.shape[0]})"
