# Oracle integration contract (Signal Manifold v0, row #29)

Status: DRAFT for review. Defines the minimal seam any oracle / tiny-expert
implements to (a) emit `SignalRecord`s and (b) plug into the fusion seam +
`manylatents` core API. This is what lets oracle #2 (ChromBPNet / Borzoi) and
later RNA-processing experts drop in behind one interface.

Owner of the shared record schema (`SignalRecord`, `Variant`, layer enum): the
signal-schema milestone (#26). This contract references those types
structurally only (`TYPE_CHECKING`) so it can land without waiting on them.

## The contract

`manylatents/dogma/oracles/__init__.py`:

```python
@runtime_checkable
class SignalOracle(Protocol):
    @property
    def oracle_id(self) -> str: ...        # stable id -> provenance + registry key
    @property
    def layers(self) -> Sequence[str]: ...       # subset of SIGNAL_LAYERS
    @property
    def cell_types(self) -> Sequence[str]: ...    # may be empty (cell-type-agnostic)
    @property
    def track_ids(self) -> Sequence[str]: ...     # join key for cross-oracle fusion

    def score_variant(self, variant, *, layers=None) -> Sequence[SignalRecord]: ...
```

Plus a decorator registry: `register_oracle(name)`, `get_oracle(name)`,
`list_oracles()`.

### Why this is the minimal surface

Derived from what the existing encoders already share, not invented:

| Contract member | Generalizes (existing encoder surface) |
|---|---|
| `score_variant() -> records` | `AlphaGenomeEncoder.predict(sequence, output_types=...)` -> per-track tensors. The **only** encoder that already emits tracks (the signal); the rest emit dense embeddings. `layers=` mirrors `output_types=`. |
| `oracle_id` | encoder `model_name` (stable id, used for weights + config keys) |
| `layers` | encoder `modality` property (single-value introspection generalized to the set of layers an oracle covers) |
| `track_ids` | AlphaGenome's `OutputType` track names -> stable column identity |
| `cell_types` | AlphaGenome's `ontology_terms` argument |
| registry (`register_oracle`/`get_oracle`/`list_oracles`) | `metrics/registry.py` decorator pattern + `algorithms/latent` `get_algorithm`/`list_algorithms` |

Deliberately **excluded** (encoder surface an oracle does *not* need):
`encode()` / dense embeddings, `embedding_dim`, `encode_batch()` +
`_tokenize_batch`/`_extract_embeddings` GPU batching, `_load_model`/`device`
lifecycle. An oracle emits *signal*, it is not an embedding backbone; forcing
it to carry that surface would be abstraction we do not need at v0.

`Protocol` (structural), not an ABC, on purpose: an oracle may **wrap** an
existing `FoundationEncoder` (an AlphaGenome adapter that calls `.predict()`
and buckets tracks into `SignalRecord`s), wrap a fresh model
(ChromBPNet/Borzoi), or be pure Python (a splicing rule-set) -- none should be
forced to subclass a shared base. `@runtime_checkable` lets the registry
sanity-check registrations.

## `manylatents/api.py` touchpoint

Core `api.run(...)` is Hydra-free: it resolves string names via Python
registries and instantiates `_target_` dicts via `importlib` (`_instantiate_target`).
The oracle path hooks in the same way, no core change required at v0:

- An oracle is a normal object. A fusion algorithm (the consumer) takes an
  oracle **config dict** (`{"_target_": "...OracleClass"}`) exactly as
  `CentralDogmaFusion` already takes `evo2_config` / `esm3_config`, and
  instantiates it lazily (Hydra `instantiate` in-pipeline, or
  `api._instantiate_target` on the API path).
- `get_oracle(name)` mirrors `get_algorithm(name)` for string-name resolution,
  so an experiment can say `oracle="alphagenome"` the way it says
  `algorithm="pca"`.

Design note (not for this row): if/when we want `api.run(..., oracle="...")`
as a first-class kwarg, the core change is a ~5-line `_resolve_oracle` helper
next to `_resolve_algorithm`. Deferred -- v0 rides the existing fusion-config
path and does not touch core.

### Fusion seam

`algorithms/fusion.py::CentralDogmaFusion` concatenates per-modality
embeddings. The signal manifold consumes `SignalRecord`s instead: an oracle
emits records per (variant, track); the fusion/aggregation step pivots them
into a per-variant feature row keyed by `(layer, cell_type, track_id)`. The
three introspection properties (`layers` / `cell_types` / `track_ids`) exist so
that pivot can build **stable, aligned columns across oracles without running
the models** -- the join key is `track_id`.

## Next-oracle shortlist

v0 is one oracle (AlphaGenome, "roll with ours"). The contract's job is to make
#2 cheap. Priority order and rationale:

1. **ChromBPNet** -- 1 bp-resolution TF-motif / accessibility corroboration.
   Emits `accessibility` + `tf` layers; smallest model, fastest to stand up,
   and gives an independent read on the exact layers AlphaGenome is weakest on
   at base-pair precision. Best first cross-check: same layers, different
   inductive bias -> a real agreement signal.
2. **Borzoi** -- RNA-seq / CAGE cross-check. Emits `rna` + `cage` (+ some
   accessibility/histone). Overlaps AlphaGenome's regulatory tracks with a
   different architecture and training set; the natural cross-oracle agreement
   test for the expression layers.
3. **First RNA-processing expert** (splicing / polyA / stability) -- starts
   filling the RNA gap the current DNA-centric stack leaves. Candidates:
   SpliceAI / Pangolin (splice), APARENT2 (polyA). Likely `cell_types = ()`
   (cell-type-agnostic) -- which is exactly why the contract makes
   `cell_types` allowed-empty rather than required.

Sequencing: (1) proves cross-oracle agreement on shared layers with minimal
cost; (2) extends it to expression; (3) opens the RNA-processing axis the
manifold currently cannot see.

## Files

- `manylatents/dogma/oracles/__init__.py` -- contract + registry
- `tests/dogma/test_oracle_contract.py` -- model-free smoke test
- `manylatents/dogma/__init__.py` -- `oracles` added to lazy `__all__`

## Open questions for #26 / César

- Confirm `SignalRecord` / `Variant` import path (assumed
  `manylatents.dogma.signal`) so the `TYPE_CHECKING` import resolves.
- `SIGNAL_LAYERS` is mirrored here as a plain tuple; should the canonical layer
  enum live in #26's schema and be re-exported, or stay duplicated with a
  sync note? (Kept minimal for now; no runtime dependency either way.)
- Is per-variant `score_variant` the right granularity, or should the contract
  also expose a batched `score_variants(iterable)` up front? Left out as
  premature -- the encoder `encode_batch` pattern shows batching can be added
  as an optional method later without changing the core contract.
