# Foundation Model Encoders

manylatents-omics provides pretrained foundation model encoders for biological sequences. All encoders inherit from `FoundationEncoder`, which provides lazy model loading, batched encoding with automatic OOM retry, and a standard `fit`/`transform` interface compatible with the manylatents experiment pipeline.

---

## Encoder Summary

| Encoder | Domain | Embedding Dim | VRAM | Architecture |
|---------|--------|---------------|------|--------------|
| [ESM3](#esm3) | Protein | 1536 | 16 GB+ | Transformer (1.4B params) |
| [Evo2](#evo2) | DNA | 1920 / 4096 / 8192 | 24 GB+ (Ampere+) | StripedHyena 2 (1B/7B/40B) |
| [Orthrus](#orthrus) | RNA | 256 / 512 | 8 GB+ | Mamba SSM |
| [AlphaGenome](#alphagenome) | DNA | 1536 / 3072 | 40 GB+ | JAX-based (DeepMind) |

---

## ESM3

**Domain:** Protein sequences (amino acids)

ESM3 is a frontier multimodal protein model from EvolutionaryScale that jointly reasons across sequence, structure, and function. The open model (`esm3-sm-open-v1`) has 1.4 billion parameters.

- **Embedding dimension:** 1536
- **VRAM:** 16 GB+
- **Pooling:** Masked mean pooling over sequence length
- **Batched inference:** True GPU batching with tokenizer padding

Key features:

- True batched forward pass (single GPU kernel per micro-batch)
- Automatic sequence truncation via `max_length`
- Loads from HuggingFace or local weights

```python
from manylatents.dogma.encoders.esm3 import ESM3Encoder

encoder = ESM3Encoder(max_length=2000)
embedding = encoder.encode("MKFGVRA")  # (1, 1536)
```

**Reference:** Hayes et al. (2024) "Simulating 500 million years of evolution with a language model"

---

## Evo2

**Domain:** DNA sequences (nucleotides)

Evo2 is a DNA language model using the StripedHyena 2 architecture, modeling DNA at single-nucleotide resolution with up to 1 million base pair context length. Available in 1B, 7B, and 40B parameter variants.

- **Embedding dimensions:** 1920 (1B), 4096 (7B), 8192 (40B)
- **VRAM:** 24 GB+ (requires Ampere or newer GPU)
- **Pooling:** Masked mean pooling over sequence length
- **Multi-layer extraction:** Extracts from multiple internal layers simultaneously

Key features:

- Multi-layer embedding extraction (default: 3 layers at 56%, 76%, 92% depth for 1B)
- Returns `dict[str, Tensor]` when multi-layer mode is active
- True batched forward pass with tokenizer padding
- OOM retry with automatic batch size halving

| Model | Parameters | Hidden Dim | Default Layers |
|-------|-----------|------------|----------------|
| `evo2_1b_base` | 1B | 1920 | blocks.14, blocks.19, blocks.23 |
| `evo2_7b` | 7B | 4096 | blocks.16 |
| `evo2_40b` | 40B | 8192 | blocks.32 |

```python
from manylatents.dogma.encoders.evo2 import Evo2Encoder

encoder = Evo2Encoder(model_name="evo2_1b_base")
result = encoder.encode("ATGAAGTTTGGCGTCCGTGCCTGA")
# Multi-layer default: result is dict with 3 layer keys
```

**Reference:** Nguyen et al. (2025) "Genome modeling and design across all domains of life with Evo 2"

---

## Orthrus

**Domain:** RNA sequences (nucleotides)

Orthrus is a Mamba SSM-based RNA foundation model. The manylatents-omics implementation is a native re-implementation compatible with mamba-ssm 2.x, avoiding version conflicts with Evo2.

- **Embedding dimensions:** 256 (4-track base), 512 (6-track large)
- **VRAM:** 8 GB+
- **Input encoding:** One-hot (A, C, G, U)
- **Pooling:** Length-aware mean pooling (respects padding)

Key features:

- Native mamba-ssm 2.x implementation (no dependency conflict with Evo2)
- Supports multi-layer intermediate capture
- Loads pretrained weights from HuggingFace

| Model | Tracks | Hidden Dim | Layers |
|-------|--------|------------|--------|
| `orthrus-base-4-track` | 4 | 256 | 8 |
| `orthrus-large-6-track` | 6 | 512 | 12 |

```python
from manylatents.dogma.encoders.orthrus_native import OrthrusNativeEncoder

encoder = OrthrusNativeEncoder()
embedding = encoder.encode("AUGCAUGCAUGCAUGC")  # (1, 256)
```

**Reference:** Fradkin et al. (2024) "Orthrus: Towards Evolutionary and Functional RNA Foundation Models"

---

## AlphaGenome

**Domain:** DNA sequences with regulatory predictions

AlphaGenome is a JAX-based genomics foundation model from Google DeepMind that predicts regulatory features at single base-pair resolution across 1 Mb context windows.

- **Embedding dimensions:** 1536 (1bp resolution), 3072 (128bp resolution)
- **VRAM:** 40 GB+
- **Framework:** JAX internally, PyTorch tensor output via `torch-jax-interop`
- **Context length:** 1,000,000 bp

Key features:

- Dual mode: embeddings (`encode`) and regulatory track predictions (`predict`)
- Chunked encoding for sequences longer than context window
- Regulatory track prediction (ATAC, CAGE, DNASE, RNA-seq, and more)
- Automatic JAX compatibility patching for older JAX versions

| Model | Resolution | Embedding Dim | Default Layer |
|-------|-----------|---------------|---------------|
| `alphagenome` | 1 bp | 1536 | `embeddings_1bp` |
| `alphagenome_128bp` | 128 bp | 3072 | `embeddings_128bp` |

```python
from manylatents.dogma.encoders.alphagenome import AlphaGenomeEncoder

encoder = AlphaGenomeEncoder()
embedding = encoder.encode("ATGAAGTTTGGCGTCCGTGCCTGA")  # (1, 1536)
predictions = encoder.predict("ATGAAGTTTGGCGTCCGTGCCTGA")  # dict of track tensors
```

**Reference:** "AlphaGenome: Foundation model for the human genome" (Google DeepMind)

---

## FoundationEncoder Base Class

All encoders inherit from `FoundationEncoder`, which extends `LatentModule` with:

- **Lazy loading:** Models are loaded on first `encode()` call, not at instantiation
- **Batched encoding:** `encode_batch()` chunks inputs into micro-batches with automatic OOM retry (halves batch size on CUDA OOM, retries without resetting)
- **fit/transform interface:** `fit()` is a no-op; `transform()` reads sequences from the datamodule and calls `encode_batch()`
- **True batched forward:** Subclasses that implement `_tokenize_batch()` and `_extract_embeddings()` get single-kernel-per-batch GPU inference instead of looped single-sample encoding

```python
# All encoders follow the same interface
encoder.fit(x)                          # no-op for pretrained models
embeddings = encoder.transform(x)       # encodes sequences from datamodule
embeddings = encoder.encode_batch(seqs) # direct batched encoding
```

## Non-coding variant benchmarks

`pip install 'manylatents-omics[noncoding]'`

```python
from manylatents.dogma import atlas
from manylatents.dogma.data import traitgym

variants = traitgym.load_variants("mendelian_traits_matched_9")   # chrom, pos, ref, alt, label, variant_id

client = atlas.create_client()                    # reads ALPHAGENOME_API_KEY
print(list(client.scorer_metadata()))             # available scorer names
atlas.fetch_atlas_scores(client, variants, ["<scorer>"], cache_dir="atlas_cache")
matrix = atlas.load_atlas_matrix("atlas_cache", "<scorer>", variants["variant_id"])
matrix.values      # (n_variants, n_tracks) float32, rows in the order given
matrix.tracks      # one row of track metadata per column
```

`fetch_atlas_scores` is resumable: finished chunks are skipped on the next
call. Variants the service cannot serve are listed by
`atlas.missing_variants(cache_dir, scorer)` and are never stored as rows of
zeros; asking `load_atlas_matrix` for one raises `AtlasScoresMissing`.
Gene-scoped scorers return one row per variant and gene, so reading them needs
`gene_reduce="maxabs"`.

AlphaGenome Atlas outputs are subject to Google DeepMind's terms of use, which
restrict them to non-commercial use and exclude training other machine-learning
models on them. Read the terms before fetching.

Locus helpers use only NumPy and pandas:

```python
from manylatents.dogma.benchmarks.loci import (
    chain_loci, genomic_separation, positional_scores, distant_neighbour_distances,
)

loci = chain_loci(variants, gap=1000)  # controls inherit their matched positive's locus
separation = genomic_separation(variants)
scores = positional_scores(variants)  # the nearest-positive score uses labels
# distance is a feature-distance matrix with infinity on its diagonal.
nearest = distant_neighbour_distances(distance, separation, exclusion=1000, k=10)
```

Sample a background from a full TraitGym pool, excluding benchmark IDs:

```python
import numpy as np

pool = traitgym.load_pool("mendelian_traits_all")
background = traitgym.sample_background(
    pool, 1000, np.random.default_rng(42), exclude=variants["variant_id"],
    match_to=variants, column="consequence",
)
print(background.attrs["shortfall"], background.attrs["shortfall_by_stratum"])
```

Omit `match_to` for uniform sampling. Matched quotas use largest-remainder
rounding; scarce strata return all available candidates without redistributing
the shortfall. Both modes return rows in pool order.

Read all available scores across local cache shards without a service call:

```python
matrix = atlas.load_cached_scores(
    ["scores/shard_000_of_002", "scores/shard_001_of_002"], "DNASE",
)
```

Empty caches are skipped. Duplicate variants or differing track metadata raise
`ValueError`; no cached scores raises `AtlasScoresMissing`. Pass `layer` or
`gene_reduce` as for `load_atlas_matrix`.

Score a variant TSV locally using an already cached AlphaGenome checkpoint:

```bash
python -m manylatents.dogma.score_tracks --variants variants.tsv --out scores \
    --fasta reference.fa --shard 0 --n-shards 2
```

`--fasta` is required. `shard_slice(n_rows, shard, n_shards)` produces contiguous
shards; `prefilter_variants(variants, fasta_path, sequence_length)` returns kept
and dropped tables, with a `reason` column on dropped rows. Input to the helper
uses canonical `chromosome` names and 1-based `pos`; the CLI adds canonical
identifiers from `chrom`, `pos`, `ref`, `alt`. FASTA access is lazy, and `--help`
does not load the model libraries.
