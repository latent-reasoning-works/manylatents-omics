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

### Hidden states at the variant site

`manylatents.dogma.alphagenome` runs AlphaGenome as a plain Haiku function
(`alphagenome-jax` extra, plus `alphagenome_research` from GitHub) and reads its
hidden state at the variant, layer by layer. `load_local()` restores a checkpoint
from the local Hugging Face cache only. `taps.DEFAULT_TRUNK_LAYERS` lists 26
trunk layers in forward order (DNA embedder, six down-resolution blocks, the
residual stream entering each of the nine attention blocks, the tower output,
seven up-resolution blocks, and the 128 bp and 1 bp output embedders);
`taps.site_states()` returns, for each, the bin that holds the site at that
layer's resolution. An SNV sits at index `length // 2 - 1` (524,287 at 1 Mb),
as for `variant.reference_interval.resize(length)`.

```bash
python -m manylatents.dogma.alphagenome.site_states --variants variants.tsv \
    --fasta reference.fa --out site_states --sequence-length 1048576 \
    --alleles all --shard 0 --n-shards 4
```

Each shard writes `states` (float16), `deltas` (float32, alt - ref) and
`features` (float32 `delta_norm`, `cosine`, `pr3`) parquet chunks, computed
from the float32 states, and resumes finished chunks. `--alleles all` also
scores the two other bases, which `pr3` (the participation ratio of the three
alternate-allele displacements, in [1, 2]) needs; `--repeat N` scores each
variant N times for a noise floor. `latent_features` holds the numpy feature
functions, shared with other models. `algorithms/latent=alphagenome_site`
(`AlphaGenomeSiteEncoder`) returns the delta at one layer, or the features
across layers, for a datamodule with `get_variants()`, from the model or from
`states_dir`.

### Finetuning and steering

`algorithms/latent=alphagenome_finetune` (`AlphaGenomeFinetune`) trains a small
set of parameters beside a frozen AlphaGenome and returns, per variant, the
edited site state, the edited alt - ref delta or a head's prediction. Every
AlphaGenome parameter stays bit-identical; a `JaxTrainer` (optax, nested in
the algorithm config) optimises only the edit and the heads.

| Intervention | What it trains |
|---|---|
| `LowRankEdit` | A LoReFT-style edit `h + R^T (W h + b - R h)` at one tap (`layer`), at all positions or only at the site's bins. `R` has orthonormal rows by construction (sign-fixed QR of a free matrix). At step 0 the edit is the identity. |
| `HeadOnly` | No edit. Heads read frozen states at one or more `readout_layers`; the states are computed once and cached. |

| Objective | Loss |
|---|---|
| `TrackHead` | A linear or one-hidden-layer head from the readout state (`source: delta`, `ref` or `alt`) to a per-variant target vector, with `mse`, `poisson` or `cosine` loss. |
| `DiffusionOperatorMatch` | Symmetric KL between the row-normalised Gaussian kernels (bandwidth 0.25 times the median distance) of the student's and a teacher's embeddings on each minibatch. |
| `JacobianPenalty` | Squared Frobenius norm of the edit's Jacobian at the edit layer, by Hutchinson probes or exactly. |
| `CovarianceIsotropy` | Off-diagonal share of the batch covariance. |

`Composite` sums objectives by their `weight`. `alphagenome_finetune_operator`
and `alphagenome_head_only` are two more configs: an operator match with a
Jacobian penalty, and a track head on frozen states.

Targets (`targets_path`) and teachers (`teacher_path`) are per-variant tables
or arrays that the caller supplies. They must not come from the labels of a
benchmark used for evaluation. For this reason `fit()` ignores its `y` unless
the config sets `targets_from: y`.

Gradients flow back from the readout layers to the edit layer. The default
length, 131,072 bp, is chosen so that a tower readout fits on one H100; this
has not yet been measured on the real model. A readout in the decoder at 1 Mb
holds 1 bp activations for backpropagation and needs an H200, or
`trainer.remat: true`, which recomputes each block in the backward pass.

### Track geometry evaluation

`manylatents.dogma.benchmarks.track_geometry.run` evaluates a variant-by-track
query matrix against named reference matrices (the same track columns), with
`None` selecting the query itself as a transductive cloud. Supply an aligned
variants table with `chrom`, `pos` and boolean `label`, the primary cloud and
transform, sensitivity transforms, and `(cloud, transform, k)` control settings.
A frozen `Settings` holds neighbour counts, PCA dimensions, resampling counts,
control seeds, locus exclusions and score orientations.

```python
import numpy as np

from manylatents.dogma.benchmarks.track_geometry import Settings, run

settings = Settings()
report, scores = run(
    query=query, clouds={"reference": reference, "transductive": None},
    variants=variants, primary_cloud="reference", primary_transform="raw",
    transforms=("raw", "zscore", "unit"),
    control_settings=(("reference", "raw", settings.primary_k),),
    random_tracks=None, rng=np.random.default_rng(settings.seed), report={},
    settings=settings,
)
```

The report includes all primary label-free scores, chromosome-weighted AUPRC
in both orientations, bootstrap errors, a within-chromosome label-permutation
null, a cloud/transform/neighbour-count sensitivity grid, row-shuffled and
Gaussian-reference controls, and magnitude and locus-confound analyses. The
permutation null gives the chance level: chromosome-weighted AUPRC for an
uninformative score sits above prevalence. `scores` contains the primary
per-variant arrays; undefined measurements are counted in the report.

An optional `random_tracks(rng)` callable supplies an alternative query and
cloud mapping for random-track controls. Pass `log` to receive progress
messages (default: `print`). This module requires the upstream geometry,
grouped-average-precision and surrogate helpers; until the dependency pin is
updated, use a development installation containing those helpers.
