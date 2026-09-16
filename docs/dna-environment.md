# The DNA environment, and why it is not `uv.lock`

The genomic arm runs in `.venv-dna`, a hand-built environment. This is
deliberate, and the repository's own `uv.lock` **must not** be synced over it.

## What the lock omits

| package | `uv.lock` | `.venv-dna` (what runs) |
|---|---|---|
| torch | 2.14.0 | **2.8.0+cu126** |
| torchvision | — | 0.23.0+cu126 |
| **evo2** | **absent** | 0.4.0 |
| **flash_attn** | **absent** | 2.8.3 |
| numpy | 2.2.6 | 1.26.4 |
| scipy | 1.14.1 | 1.11.4 |
| transformers | 4.48.1 | 4.48.1 |
| manylatents | 0.1.7 | 0.1.7 |

`evo2` and `flash_attn` — the two packages the arm cannot run without — appear
nowhere in the lock. They are installed from sources `uv` does not resolve.

## Why syncing would break it

`flash_attn 2.8.3` is a compiled CUDA extension built against torch 2.8/cu126.
Replacing torch with the locked 2.14.0 leaves the extension linked against an
ABI that is gone, and `evo2`/`vortex` import it on the attention path. The
failure would surface as an import or CUDA error deep inside a model forward,
far from the change that caused it.

This is the opposite situation to the text campaign, where `uv sync --frozen`
was exactly right: there, the lock was authoritative and the deployment had
drifted from it. Here the deployment is authoritative and the lock is
incomplete.

## The pin

`requirements-dna.lock` is a full freeze of the environment that produced the
Evo 2 results — 250 packages, captured from Tamia after the arm ran end to end
(job 465817: ClinVar loaded, Evo2-1B cached at 1024 bp, 3.16 GB peak,
0.083 s/batch, zero OOM events).

Rebuild from it rather than from `uv.lock`:

```bash
uv pip install --python .venv-dna/bin/python -r requirements-dna.lock
```

`flash_attn` may need `--no-build-isolation` so it compiles against the torch
already present rather than pulling its own.

## What is still unpinned

`vortex` does not appear in the freeze under that name; it arrives as a
dependency of `evo2` and its distribution name differs. Anyone rebuilding this
environment should confirm `import vortex.model.model` succeeds before trusting
the result — it is the module the trainability fix depends on
(`utils.py:102`, `torch.inference_mode` around `load_checkpoint`).
