# Design: ClinVar Geometric Pipeline

## Context

The ClinVar database contains clinical interpretations of genetic variants (pathogenic, benign, VUS). For geometric analysis of foundation model embeddings, we need to:
1. Extract aligned DNA/RNA/Protein sequences for each variant
2. Encode sequences with Evo2 (DNA), Orthrus (RNA), ESM3 (Protein)
3. Fuse embeddings from multiple modalities
4. Compute geometric metrics (PR, LID, TSA) stratified by pathogenicity

## Goals / Non-Goals

### Goals
- Hydra-native experiment definition via config composition
- Shop-compatible SLURM submission
- Reusable components for multi-modal embedding fusion
- Clean separation: DataModules load data, LatentModules transform it

### Non-Goals
- Real-time ClinVar API integration (batch preprocessing is sufficient)
- Novel fusion architectures (simple concat/weighted fusion first)
- Custom metric implementations (reuse manylatents metric registry)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Component Architecture                          │
└─────────────────────────────────────────────────────────────────────────┘

DataModules (load data):
┌──────────────────────┐    ┌─────────────────────────────┐
│  ClinVarDataModule   │    │ PrecomputedEmbeddingsData   │
│  get_sequences()     │    │ get_embeddings()            │
│  get_labels()        │    │ get_labels()                │
│  get_variant_ids()   │    │ get_variant_ids()           │
└──────────────────────┘    └─────────────────────────────┘
         ↓                              ↓
LatentModules (transform):
┌──────────────────────┐    ┌─────────────────────────────┐
│    BatchEncoder      │    │   PrecomputedFusionModule   │
│  encode sequences    │    │   fuse channels             │
│  save to .pt files   │    │   (concat/weighted/attn)    │
└──────────────────────┘    └─────────────────────────────┘
         ↓                              ↓
         └──────────────┬───────────────┘
                        ↓
              manylatents metrics
              (PR, LID, TSA)
```

## Data Flow

### Phase 1: Sequence Encoding (GPU jobs, parallel)
```
ClinVarDataModule → BatchEncoder(Evo2)    → evo2.pt     (DNA, 1920-dim)
ClinVarDataModule → BatchEncoder(Orthrus) → orthrus.pt  (RNA, 256-dim)
ClinVarDataModule → BatchEncoder(ESM3)    → esm3.pt     (Protein, 1536-dim)
```

**Note**: All three encoders now work in the same environment. OrthrusEncoder was
re-implemented to use mamba-ssm 2.x API, eliminating the version conflict with Evo2.

### Phase 2: Fusion + Metrics (CPU job)
```
PrecomputedEmbeddingsDataModule(evo2.pt, orthrus.pt, esm3.pt)
         ↓
MergingModule(strategy=concat)  → 3712-dim fused embeddings
         ↓
manylatents.main with metrics=[pr, lid, tsa]
```

## Decisions

### Decision 1: Precomputed Embeddings as DataModule

**What**: `PrecomputedEmbeddingsDataModule` loads HDF5/PT files with multiple embedding channels.

**Why**:
- DataModules are the standard manylatents interface for data
- Allows fusion module to be agnostic to where embeddings came from
- Enables caching and reuse across experiments

**Interface**:
```python
class PrecomputedEmbeddingsDataModule(LightningDataModule):
    def __init__(
        self,
        embeddings_dir: Path,
        channels: List[str],  # ['dna/evo2', 'protein/esm3', 'rna/orthrus']
        labels_path: Optional[Path] = None,
    ):
        ...

    def get_embeddings(self) -> Dict[str, Tensor]:
        """Return dict of channel_name -> embeddings tensor."""

    def get_labels(self) -> np.ndarray:
        """Return pathogenicity labels."""

    def get_tensor(self) -> Tensor:
        """Return concatenated embeddings for LatentModule compatibility."""
```

### Decision 2: Fusion as LatentModule

**What**: `PrecomputedFusionModule` is a LatentModule that fuses multi-channel embeddings.

**Why**:
- LatentModules are the standard manylatents interface for embedding transformations
- Allows composition with downstream algorithms (PCA, UMAP, etc.)
- Enables different fusion strategies via config

**Interface**:
```python
class PrecomputedFusionModule(LatentModule):
    def __init__(
        self,
        strategy: str = "concat",  # 'concat', 'weighted', 'attention'
        channels: Optional[List[str]] = None,  # Which channels to fuse
        weights: Optional[Dict[str, float]] = None,  # For weighted fusion
        normalize: bool = False,
        **kwargs,
    ):
        ...

    def transform(self, x: Tensor) -> Tensor:
        """Fuse embeddings from datamodule.get_embeddings()."""
        embeddings = self.datamodule.get_embeddings()
        # Apply fusion strategy
        return fused
```

### Decision 3: Metrics via Experiment Config

**What**: Use existing manylatents metric registry, configured via experiment YAML.

**Why**:
- Metrics already implemented in manylatents
- No new code needed
- Sweepable via Hydra multirun

**Example config**:
```yaml
# experiment/clinvar/geometric_analysis.yaml
defaults:
  - override /data: precomputed_embeddings
  - override /algorithms/latent: precomputed_fusion

metrics:
  - participation_ratio
  - local_intrinsic_dimensionality
  - tangent_space_alignment

# Sweep over dimensionality
algorithms:
  latent:
    n_components: 50  # Or sweep: 5, 50, 100
```

### Decision 4: HDF5 Schema

**What**: Standardized HDF5 structure for multi-channel embeddings.

```
clinvar_embeddings.h5
├── variant_ids: str[N]           # Alignment key
├── labels: int[N]                # 0=benign, 1=pathogenic
├── channels/
│   ├── dna_evo2: float32[N, 2048]
│   ├── rna_orthrus: float32[N, 256]
│   └── protein_esm3: float32[N, 1536]
└── metadata/
    ├── gene_symbol: str[N]
    └── clinical_significance: str[N]
```

**Why**:
- HDF5 supports lazy loading (memory efficient)
- Hierarchical structure for channels
- Metadata for stratified analysis

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| ~~Orthrus env conflict~~ | **RESOLVED**: Re-implemented OrthrusEncoder with mamba-ssm 2.x |
| Large embedding files | Use HDF5 compression, lazy loading |
| Channel misalignment | Align by variant_id, validate on load |

## Open Questions

1. **Fusion strategies beyond concat?**
   - Start with concat, add weighted/attention later if needed

2. **Dimensionality sweep values?**
   - Start with 5, 50, 100; tune based on neighborhood structure

3. **Stratification granularity?**
   - Global + pathogenic/benign; per-gene optional
