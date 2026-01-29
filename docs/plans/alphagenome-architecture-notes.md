# AlphaGenome Architecture Triage Notes

## Model Architecture

```
DNA Sequence (B, S, 4)
        │
        ▼
┌─────────────────┐
│ SequenceEncoder │  Conv layers with pooling (S → S/128)
└────────┬────────┘
         │ trunk: (B, S/128, D)
         ▼
┌─────────────────┐
│ TransformerTower│  9 transformer blocks with pairwise attention
└────────┬────────┘
         │ trunk: (B, S/128, D), pair: (B, S/2048, S/2048, F)
         ▼
┌─────────────────┐
│ SequenceDecoder │  Upsampling back to (B, S, D)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ OutputEmbedder  │  Creates final embeddings
└────────┬────────┘
         │
         ▼
    Prediction Heads (ATAC, CAGE, RNA_SEQ, etc.)
```

## Embedding Dimensions

| Resolution | Shape | Dimension |
|------------|-------|-----------|
| embeddings_1bp | `(B, S, 1536)` | 1536 |
| embeddings_128bp | `(B, S//128, 3072)` | 3072 |
| embeddings_pair | `(B, S//2048, S//2048, 128)` | 128 |

## API Methods

### Factory Methods
- `dna_model.create_from_huggingface(model_version, *, organism_settings=None, device=None)`
- `dna_model.create_from_kaggle(model_version, *, organism_settings=None, device=None)`

### Prediction Methods
- `model.predict_sequence(sequence, *, organism, requested_outputs, ontology_terms, interval=None) -> Output`
- `model.predict_interval(interval, *, organism, requested_outputs, ontology_terms) -> Output`
- `model.predict_variant(interval, variant, *, organism, requested_outputs, ontology_terms) -> VariantOutput`

### OutputType Enum Values
- `ATAC` - ATAC-seq (chromatin accessibility)
- `CAGE` - CAGE (transcription start sites)
- `CHIP_HISTONE` - ChIP-seq histone marks (128bp resolution)
- `CHIP_TF` - ChIP-seq transcription factors (128bp resolution)
- `CONTACT_MAPS` - Hi-C contact maps
- `DNASE` - DNase-seq (chromatin accessibility)
- `PROCAP` - PRO-cap (nascent transcription)
- `RNA_SEQ` - RNA-seq (gene expression)
- `SPLICE_JUNCTIONS` - Splice junctions
- `SPLICE_SITES` - Splice site classification
- `SPLICE_SITE_USAGE` - Splice site usage

## Embedding Extraction Strategy

The internal `_predict` function returns a predictions dict that includes `embeddings_1bp`:

```python
predictions = apply_fn(params, state, sequences, organism_indices)
# predictions['embeddings_1bp'] is (B, S, 1536)
```

However, the public API (`predict_sequence`) does NOT expose raw embeddings - it only returns
the prediction track data (TrackData objects).

### Options for Embedding Extraction

1. **Mean pool embeddings_1bp** - Access via internal `_predict` call, mean pool over sequence
2. **Use Output.embeddings attribute** - Check if Output has embeddings (TBD)
3. **Custom forward pass** - Create our own haiku forward that returns embeddings

### Recommended Approach

For `encode()`: We need to access the internal predictions to get `embeddings_1bp`.
The challenge is that `predict_sequence` doesn't expose this.

**Solution**: Use low-level `_predict` or create a custom apply function that returns embeddings.

For `predict()`: Use the public API `predict_sequence` with all requested output types,
then convert TrackData objects to tensors.

## MODELS Dict Values

```python
MODELS = {
    "alphagenome": {
        "default_layer": "embeddings_1bp",  # Use 1bp resolution
        "embedding_dim": 1536,
        "context_length": 1_000_000,
    },
    "alphagenome_128bp": {
        "default_layer": "embeddings_128bp",  # Coarser resolution
        "embedding_dim": 3072,
        "context_length": 1_000_000,
    },
}
```

## Implementation Notes

1. **JAX/PyTorch interop**: Use `torch_jax_interop.jax_to_torch()` for conversion
2. **Device handling**: JAX manages its own devices; we convert to PyTorch and move to target device
3. **One-hot encoding**: Model expects `(B, S, 4)` one-hot encoded sequence
4. **Organism**: Default to `Organism.HOMO_SAPIENS`
5. **Ontology terms**: Required for some outputs, can be None for embeddings
