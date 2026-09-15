# Staging Evo 2 40B onto Tamia

Record for plan Task 1, `docs/superpowers/plans/2026-09-15-evo2-genomic-arm.md`.
Performed 2026-09-15.

## Source

`mila:/network/weights/evo2/evo2_40b/`, read-only, owned by `mila_operators`.
The checkpoint ships as two shards because of HuggingFace file-size limits,
alongside `config.json`, `README.md`, `.gitattributes` and a `.git` pointer.

| file | bytes |
|---|---|
| `evo2_40b.pt.part0` | 41,126,745,847 |
| `evo2_40b.pt.part1` | 41,126,745,847 |
| metadata (4 files) | 2,768 |

## Destination

`tamia:/scratch/c/cesarmvc/merging-dogma/models/evo2_40b/`.

Deliberately **not** under `models/huggingface/hub/`. The 40B is a raw two-part
`.pt` loaded through `local_path`, not an HF snapshot; placing it in the hub tree
invites `from_pretrained` to misresolve it.

## Transfer

`rsync -a --partial --stats`, Mila → Tamia, 2026-09-15 13:58–14:26 local.

```
Total transferred file size: 82,253,494,462 bytes
sent 82,273,576,309 bytes  received 133 bytes  49,697,116.55 bytes/sec
```

82,253,491,694 bytes of shards plus 2,768 of metadata accounts for the total
exactly.

## Integrity

Sizes were checked first, but sizes cannot tell a corrupted byte from a correct
one — a point astra raised in review, correctly. sha256 was therefore computed
independently on both sides and compared.

| file | sha256 | source | destination |
|---|---|---|---|
| `evo2_40b.pt.part0` | `3b74fa4e6158d49265e3e270ba8869390d064358f8bf3d2af0b3e1772728f485` | ✓ | ✓ |
| `evo2_40b.pt.part1` | `bdc4a76e0f23f8295e7061c2f0deff24f723bd916dc4cdc4d9216cac9c2d49d5` | ✓ | ✓ |
| `evo2_40b.pt` (merged) | `dd299612b1c1cdded0dfdcaf4d16f98fc97458261d80f4d662429f0ccb316bc3` | n/a | ✓ |

Both shards match byte-for-byte. The source hashes are staged beside the weights
as `source.sha256`; point `EVO2_40B_SOURCE_SHA256` at that file and
`stage_evo2_40b.sh --verify-only` will re-check them and fail on mismatch.

## The merge, which the plan omitted

`load_checkpoint` takes one file. `evo2.models.load_evo2_model` concatenates the
shards before loading, so **staging is not complete at transfer** — the merge is
required. It is a plain byte concatenation in part order:

```
cat evo2_40b.pt.part0 evo2_40b.pt.part1 > evo2_40b.pt   # 82,253,491,694 bytes
```

Shards are kept rather than removed as the package does. Re-merging costs
minutes; re-transferring costs 82 GB. The cost is 164 GB on scratch, which went
543 G → 696 G of a 2.0 T filesystem.

## Loading

Set `EVO2_WEIGHTS_DIR=/scratch/c/cesarmvc/merging-dogma/models` and
`Evo2Encoder` resolves `<root>/evo2_40b/evo2_40b.pt`. It resolves the **file**,
never the directory: `Evo2` passes `local_path` straight to `torch.load`.

## Not yet verified

The 40B has not been loaded. 82.25 GB of bf16 parameters exceeds one 80 GB H100,
so it needs one of Tamia's H200 nodes (141 GB, `gpubase_bynode_b1`). Whether the
remaining headroom supports a usable ClinVar window is Task 5's measurement.
