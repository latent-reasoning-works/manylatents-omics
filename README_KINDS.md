# Data Kinds

**Typed internal data representations for omics workflows.**

This defines the schema seam between data loaders and ops/algorithms. Each kind carries its own structural semantics (named dimensions, coordinates, required fields), so ops can read and validate structure instead of guessing axes by convention.

## Directory Structure

```
manylatents/singlecell/data/
├── kinds.py                    # Typed kinds (LabeledArray, SparseGraph, Trajectory)
├── adapters.py                 # Generic: AnnData → LabeledArray converter
├── tenx.py                     # Specific: 10x h5ad loader
├── anndata.py                  # (existing loaders)
├── cellxgene_census.py
└── ...
```

Added to the existing loader structure without reorganization.

## The Problem We're Solving

- ❌ **AnnData as internal type**: Can't cleanly represent trajectories, time-series, or graphs.
- ❌ **Bare numpy arrays + positional convention**: Dims get reordered, and code has to *guess* what axis 0 means.
- ✅ **Typed kinds with named dims**: Structure is self-describing. Ops read dims, never guess them.

## The Three Kinds

### LabeledArray

**xarray DataArray with named dimensions.** The primary kind for cell×gene matrices and other labeled array data.

**Required dims:** `cell`, `gene`

**Optional dims:** `time` (for time-series data)

**Attributes:** Carries domain metadata (e.g., `genome`, `gene_ids`) in `.attrs`.

#### Construction

```python
import xarray as xr
from manylatents.singlecell.data.kinds import LabeledArray

# From raw numpy
data = np.random.rand(1000, 2000)  # 1000 cells × 2000 genes
da = xr.DataArray(
    data,
    dims=["cell", "gene"],
    coords={
        "cell": cell_ids,
        "gene": gene_names,
    },
    attrs={"genome": "GRCh38"}
)
kind = LabeledArray(da)
kind.validate()
```

#### Usage in Ops

```python
def my_op(kind: LabeledArray) -> LabeledArray:
    """Op that requires 'time' dimension."""
    if "time" not in kind._da.dims:
        raise ValueError("This op requires a 'time' dimension")
    
    # Operate on kind._da
    result = kind._da.mean(dim="time")
    return LabeledArray(result)
```

#### Serialization

```python
# Write to zarr
kind.serialize("data.zarr")

# Load back (validation runs on read)
loaded = LabeledArray.load("data.zarr")
loaded.validate()  # Passes if structure is intact
```

#### Time-Series Example

```python
# 1000 cells × 2000 genes × 10 timepoints
data = np.random.rand(1000, 2000, 10)
da = xr.DataArray(
    data,
    dims=["cell", "gene", "time"],
    coords={
        "cell": cell_ids,
        "gene": gene_names,
        "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
)
kind = LabeledArray(da, required_dims={"cell", "gene", "time"})
kind.validate()
```

### SparseGraph

**Stub for typed graph representations.** Will be fleshed out for RITINI-style regulatory networks and other graph algorithms.

**API (current stub):**

```python
from manylatents.singlecell.data.kinds import SparseGraph

# Construct
graph = SparseGraph(edge_list=[(source, target, weight), ...])

# Validate (no-op in stub)
graph.validate()

# Serialize/load (not yet implemented)
graph.serialize("path.zarr")
loaded = SparseGraph.load("path.zarr")
```

**Planned features (M1+):**
- Edge list → sparse adjacency matrix conversion
- Node/edge metadata
- Serialization to sparse zarr format

### Trajectory

**Stub for typed trajectory representations.** Will be fleshed out for MIOFlow-style cell fate trajectories and temporal pseudotime paths.

**API (current stub):**

```python
from manylatents.singlecell.data.kinds import Trajectory

# Construct
paths = [
    ["cell_1", "cell_2", "cell_3"],
    ["cell_4", "cell_5"],
]
traj = Trajectory(paths=paths)

# Validate (no-op in stub)
traj.validate()

# Serialize/load (not yet implemented)
traj.serialize("path.zarr")
loaded = Trajectory.load("path.zarr")
```

**Planned features (M1+):**
- Variable-length path representation
- Pseudotime values and uncertainty
- Branching structure (which paths connect)

## AnnData Adapter

The **only place** AnnData is used internally. Located in `manylatents/singlecell/data/adapters.py`. Conversion happens at the edge:

```python
from manylatents.singlecell.data.adapters import anndata_to_labeled_array

# Load h5ad
import scanpy as sc
adata = sc.read_h5ad("data.h5ad")

# Convert to typed kind
kind = anndata_to_labeled_array(adata, use_raw=False)
kind.validate()
```

Downstream code never touches AnnData — only the typed kind.

## 10x Genomics Loader

Located in `manylatents/singlecell/data/tenx.py`. Loads 10x h5ad files and converts to `LabeledArray`:

```python
from manylatents.singlecell.data import Gemonics

# Load and convert
loader = Gemonics("filtered_feature_bc_matrix.h5ad")
kind = loader.kind

# Use kind
kind.validate()
kind.serialize("output.zarr")

# Load later
loaded = Gemonics.load("output.zarr")
```

## Example Ops

Located in `tests/singlecell/test_op/example_ops.py`. Demonstrates how ops validate required dims:

### temporal_analysis

Requires `time` dimension; fails cleanly without it.

```python
from tests.singlecell.test_op.example_ops import temporal_analysis

kind = LabeledArray.load("data.zarr")

# If kind has time dim: succeeds
if "time" in kind._da.dims:
    result = temporal_analysis(kind)

# If kind doesn't have time dim: raises ValueError with clear message
```

### basic_filter

Works on any `LabeledArray`. No special dim requirements.

```python
from tests.singlecell.test_op.example_ops import basic_filter

kind = LabeledArray.load("data.zarr")
filtered = basic_filter(kind, min_expression=0.1)
```

## Adding a New Kind

1. **Define the class** in `manylatents/singlecell/data/kinds.py`:

```python
class MyKind(Kind):
    """Description."""
    
    def __init__(self, data, **kwargs):
        self._data = data
    
    def validate(self) -> None:
        """Check structure. Raise on failure."""
        if not self._is_valid():
            raise ValueError("Invalid structure")
    
    def serialize(self, path: str) -> None:
        """Write to disk."""
        ...
    
    @classmethod
    def load(cls, path: str) -> "MyKind":
        """Load from disk and validate on read."""
        obj = cls(...)
        obj.validate()
        return obj
```

2. **Write tests** in `tests/test_kinds.py`:

```python
class TestMyKind:
    def test_round_trip(self):
        """Construct → serialize → load → validate → identical."""
        ...
    
    def test_validate_rejects_malformed(self):
        """Malformed data fails validation on load."""
        ...
```

3. **Update this README** with:
   - What the kind represents
   - Required fields/dims
   - Usage examples
   - Serialization format

## Round-Trip Guarantees

Each kind must satisfy:

```
Construct → validate()
           ↓
        serialize(path)
           ↓
        load(path) → validate()  ← validation runs on read
           ↓
        identical to original
```

This ensures:
- Malformed data is rejected on load, not silently accepted
- Named dims survive slicing/transposing
- Metadata (coords, attrs) is preserved

## Testing

Run the test suite:

```bash
pytest tests/test_kinds.py -v
```

Expected coverage:
- **Round-trip tests**: Each kind's construct→serialize→load→validate cycle
- **Rejection tests**: Malformed kinds fail on load
- **Op validation tests**: Ops can require dims and fail cleanly without them

## Coordination with Downstreams

**Brian (manylatents#269)**: Model ops consume these kinds. Lock the `LabeledArray` required dims (`cell`, `gene`, optional `time`) before building against it.

**MIOFlow/RITINI**: Will flesh out `Trajectory` and `SparseGraph` stubs. Shape of ops will be finalized when implementations land.

---

*Last updated: 2026-06-03*
