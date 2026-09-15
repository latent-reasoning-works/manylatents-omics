# Can Evo 2 be trained through the diffusion-operator loss?

**Yes.** Evo2-1B trains, at 4.1 GB peak for batch 4 x 128 tokens on one H100.
The obstacle was never the architecture; it was where the weights are born.

Spike jobs `465429` (round 1, negative) and `465441` (round 2, positive) on
Tamia, 2026-09-15.

## The lever that wasn't

An earlier brief, and the arm's design doc, said the blocker was a config flag:
`vortex/model/model.py:135`, `self.inference_mode = config.get("inference_mode", True)`
— "a config default, not a constraint". `evo2/configs/evo2-1b-8k.yml` sets it
True on line 60.

That is wrong. **`self.inference_mode` is assigned at line 135 and never read
again** — one grep hit in the entire file. Round 1 patched the config to False
and got:

```
inference_mode as shipped: True
built OK
is_inference straight after load: True
RuntimeError: Inference tensors cannot be saved for backward
  at vortex/model/model.py:394 in compute_filter
```

Identical to the failure the flag was supposed to explain.

## Where the inference tensors actually come from

`vortex/model/utils.py:102`. `load_checkpoint` wraps `torch.load` and the weight
copy in `with torch.inference_mode():`, so every loaded weight is an inference
tensor permanently. `requires_grad` is True on all 1.108B parameters and means
nothing: an inference tensor cannot be saved for backward whatever its flag says.

This also explains why the original three blockers looked like three. Removing
`Evo2.forward`'s `torch.no_grad()` and its detaching hook were both necessary
and neither was sufficient, because the poison is in the weights rather than in
the forward pass.

## What works

PyTorch's own error names the fix. Re-materialise every parameter **and buffer**
as a normal clone, outside inference mode, after `load_checkpoint` returns:

```python
for module in model.modules():
    for name, param in list(module._parameters.items()):
        if param is not None:
            module._parameters[name] = torch.nn.Parameter(
                param.detach().clone(), requires_grad=param.requires_grad
            )
    for name, buffer in list(module._buffers.items()):
        if buffer is not None:
            module._buffers[name] = buffer.detach().clone()
```

Buffers are not optional. `compute_filter` evaluates
`(residues[..., None] * (log_poles * self.t).exp()).sum(1)`, and `self.t` is a
buffer — a single inference operand poisons the graph even with clean parameters.
Round 2 converted **265 parameters and 4 buffers**, after which no parameter
reported `is_inference`.

## Measured

| quantity | value |
|---|---|
| parameters re-materialised | 265 |
| buffers re-materialised | 4 |
| parameters still inference afterwards | 0 |
| hidden states at `blocks.14.mlp.l3` | `(4, 128, 1920)`, bfloat16, `requires_grad=True` |
| parameters receiving finite non-zero gradients | **159 / 265** |
| operator diagonal minimum | 3.458e-01 |
| peak memory, forward | 4.1 GB |
| peak memory, after backward | 4.1 GB |

159 of 265 is the expected number, not a shortfall. The loss is taken on the
output of `blocks.14`, so blocks 0-14 and the embedding are upstream of it and
receive gradients; blocks 15-24 and the head are downstream and correctly
receive none. Gradient norms are healthy and span the model:
`embedding_layer.weight` 1.6e+01, `blocks.0.filter.short_filter_weight` 3.3e+01,
`blocks.0.filter.h` 9.5e+00.

The operator diagonal minimum of 0.346 confirms the Gaussian self-affinity
survives — the property the minibatch proof needs and that zeroing the diagonal
would destroy.

## Consequences

- Task 4 is a small, local change to `Evo2Encoder`: a `trainable=True` path that
  clones after load, runs the forward outside `no_grad`, and hooks without
  detaching. No fight with vortex, no fork.
- The arm does not fall back to convergence-only. Arms 2-5 of the ladder are live.
- **4.1 GB at 128 tokens leaves enormous headroom on an 80 GB H100**, but this
  says nothing about ClinVar windows one to two orders of magnitude longer.
  StripedHyena's memory profile is not a transformer's. Task 5 measures it.
- **Closed 2026-09-15 by job `465469` on Tamia.** The open question was whether
  a cloned model's frozen forward reproduces an unmodified model's numbers,
  since every teacher cache depends on the default path being untouched. The
  test builds both encoders on the same sequences and compares pooled outputs:
  `test_frozen_path_is_numerically_untouched` PASSED. So did
  `test_operator_loss_reaches_model_parameters` and the six others — 8 passed in
  36m30s, 6.2 GB peak across three model loads in one process.

  The run also exposed a prerequisite nothing had tested. Tamia's `.venv-dna`
  carried `manylatents 0.1.0`, installed editable from a stale sibling checkout
  with no `manylatents/algorithms/latent/foundation_encoder.py`, so **every**
  `manylatents.dogma` import failed — Evo 2, Orthrus, ESM3 and the ClinVar
  loader alike. The spike had passed only because it bypassed the package and
  drove `evo2`/`vortex` directly. Upgraded to 0.1.7 with `uv`; Tasks 5-8 would
  otherwise have died on import.
