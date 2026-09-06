"""
LabeledArray: an xarray DataArray with named dimensions.
"""

import logging
from dataclasses import dataclass

import xarray as xr

from .base import Kind

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class LabeledArray(Kind):
    """xarray DataArray with named dimensions."""

    da: xr.DataArray
    provenance: tuple[str, ...] = ()

    def __post_init__(self):
        # frozen: bypass the immutability guard to normalize provenance to a tuple
        object.__setattr__(self, "provenance", tuple(self.provenance))
        self.validate()

    def validate(self) -> "LabeledArray":
        if not isinstance(self.da, xr.DataArray):
            raise ValueError("LabeledArray must wrap a DataArray")
        if self.da.size == 0:
            raise ValueError(f"LabeledArray is empty (shape {self.da.shape})")
        return self

    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "LabeledArray":
        missing_dims = [d for d in dims if d not in self.da.dims]
        if missing_dims:
            raise ValueError(f"requires dims {missing_dims}; got {tuple(self.da.dims)}")

        # Code can be removed if time is decided to be a dim rather than a coord
        missing_coords = [c for c in coords if c not in self.da.coords]
        if missing_coords:
            raise ValueError(f"requires coords {missing_coords}; got {tuple(self.da.coords)}")
        return self

    def tagged(self, op_name: str) -> "LabeledArray":
        """Return a copy with ``op_name`` appended to the provenance trail.

        Part of the structural ``Kind`` protocol this package satisfies. The
        consumer is an op registry ONE LAYER UP — it appends to this trail as it
        runs each op, and the trail is the state persisted when that system saves
        itself. Named by ROLE rather than by project on purpose: the consumer that
        this was written against is retired and its successor is proprietary, so a
        name here would be either wrong or one nobody outside can read.

        Immutable: the original is untouched.
        """
        return LabeledArray(self.da, self.provenance + (op_name,))

    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".zarr"):
            raise ValueError(f"path must end in .zarr, got {path!r}")
        return str(path)

    def serialize(self, path: str) -> None:
        path = self._normalize(path)
        logger.info(f"Serializing {type(self).__name__} to {path}")
        # provenance rides in attrs (only when non-empty) so the op trail survives
        # the round-trip, reusing xarray's attrs-preservation — no side channel.
        da = self.da.assign_attrs(provenance=list(self.provenance)) if self.provenance else self.da
        da.to_zarr(path, mode="w")

    @classmethod
    def load(cls, path):
        path = cls._normalize(path)
        da = xr.open_dataarray(path, engine="zarr")
        # pull provenance back out of attrs so it doesn't linger as a domain attr
        provenance = tuple(da.attrs.pop("provenance", ()))
        return cls(da, provenance=provenance)  # validate called from __post_init__

    def __repr__(self) -> str:
        return (
            f"LabeledArray(dims={list(self.da.dims)}, shape={self.da.shape}, "
            f"provenance={self.provenance})"
        )
