"""
Typed internal data representations (kinds).

Each kind carries its own structural semantics (dims, labels, coords).
This ensures ops can read and validate structure instead of guessing.
"""

import logging
from abc import ABC, abstractmethod

import xarray as xr
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class Kind(ABC):
    """Base class for all data kinds.

    Each kind owns its own structural semantics *and* its own persistence:
    subclasses must implement ``validate``, ``serialize``, and ``load`` for the
    storage format appropriate to that kind. The base declares only the contract.
    """

    @abstractmethod
    def validate(self) -> None:
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


class LabeledArray(Kind):
    """xarray DataArray with named dimensions."""
    
    def __init__(self, da: xr.DataArray): self._da = da
        
    def validate(self) -> None:
        if not isinstance(self._da, xr.DataArray):
            raise ValueError("LabeledArray must wrap a DataArray")
        return self
    
    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "LabeledArray":
        missing_dims = [d for d in dims if d not in self._da.dims]
        if missing_dims:
            raise ValueError(f"requires dims {missing_dims}; got {tuple(self._da.dims)}")
        
        # Code can be removed if time is decided to be a dim rather than a coord
        missing_coords = [c for c in coords if c not in self._da.coords]
        if missing_coords:
            raise ValueError(f"requires coords {missing_coords}; got {tuple(self._da.coords)}")
        return self

    def serialize(self, path: str) -> None:
        logger.info(f"Serializing {type(self).__name__} to {path}")
        self._da.to_zarr(path, mode="w")

    @classmethod
    def load(cls, path):
        da = xr.open_dataset(path, engine="zarr")["data"]
        return cls(da).validate()

    @property
    def da(self) -> xr.DataArray:
        return self._da
    
    def __repr__(self) -> str:
        return f"LabeledArray(dims={list(self._da.dims)}, shape={self._da.shape})"

# TODO: flesh out
# So far: container of an edge list and number of nodes
class SparseGraph(Kind):
    """torch_geometric Data graph."""

    def __init__(self, data: Data): self._data = data

    def validate(self) -> None:
        if not isinstance(self._data, Data):
            raise ValueError("SparseGraph must wrap a Data")
        return self

    def require(self, *attrs: str) -> "SparseGraph":
        missing_attrs = [a for a in attrs if a not in self._data]
        if missing_attrs:
            raise ValueError(f"requires attrs {missing_attrs}; got {tuple(self._data.keys())}")
        return self

    def serialize(self, path: str) -> None:
        import torch
        logger.info(f"Serializing {type(self).__name__} to {path}")
        torch.save(self._data, path)

    @classmethod
    def load(cls, path):
        import torch
        data = torch.load(path, weights_only=False)
        return cls(data).validate()

    @property
    def data(self) -> Data:
        return self._data

    def __repr__(self) -> str:
        return f"SparseGraph(num_nodes={self._data.num_nodes}, num_edges={self._data.num_edges})"
    
# Storage method up to change
class TrajectoryXXX(Kind):
    def __init__(self):
        pass
    
    def validate(self) -> None:
        pass

    def require(self,) -> "TrajectoryXXX":
        pass

    def serialize(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path):
        pass

    @property
    def data(self): # -> "datatype"
        pass

    def __repr__(self) -> str:
        pass
    
