"""Spike: can Evo2-1B be trained through the diffusion-operator loss?

Not code we keep. The question is whether disabling vortex's `inference_mode`
turns the model's tensors into ordinary autograd tensors, because three separate
blockers were found on an earlier H100 run:

  1. Evo2.forward wraps the model call in `with torch.no_grad():`
  2. its embedding hook stores `output.detach()`
  3. bypassing both, backward fails with
     "Inference tensors cannot be saved for backward" at
     vortex/model/model.py:394 in compute_filter

The lever is vortex/model/model.py:135 --
`self.inference_mode = config.get("inference_mode", True)` -- a config default,
not a constraint. `evo2/configs/evo2-1b-8k.yml` sets it True on line 60.

This bypasses Evo2.forward entirely: it builds StripedHyena from a patched
config, loads the same checkpoint, registers a hook that does NOT detach, and
runs backward through the operator objective the arm actually uses.
"""

import os
import traceback

import torch
import yaml

LAYER = "blocks.14.mlp.l3"      # 56% of 25 blocks; build-and-flush rules out the end
MODEL = "evo2_1b_base"


def head(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


head("environment")
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(0)
    print(f"gpu {prop.name} | {prop.total_memory / 1e9:.1f} GB")

head("patch the config")
import evo2
from evo2.utils import CONFIG_MAP
from vortex.model.model import StripedHyena
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import dotdict, load_checkpoint

config_path = os.path.join(os.path.dirname(evo2.__file__), CONFIG_MAP[MODEL])
raw = yaml.load(open(config_path), Loader=yaml.FullLoader)
print(f"config {config_path}")
print(f"inference_mode as shipped: {raw.get('inference_mode')!r}")
raw["inference_mode"] = False
config = dotdict(raw)

weights = os.environ.get("EVO2_1B_PT")
if not weights or not os.path.exists(weights):
    raise SystemExit(f"set EVO2_1B_PT to the merged checkpoint; got {weights!r}")

head("build with inference_mode disabled")
try:
    model = StripedHyena(config)
    load_checkpoint(model, weights)
    model.eval()
    print("built OK")
    print("is_inference straight after load:", next(model.parameters()).is_inference())

    # The escape hatch. load_checkpoint ran under torch.inference_mode(), so the
    # weights are inference tensors; cloning them OUTSIDE that context yields
    # ordinary autograd tensors. Buffers too -- compute_filter multiplies
    # log_poles by self.t, and one inference operand poisons the whole graph.
    cloned_params = cloned_buffers = 0
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None:
                continue
            module._parameters[name] = torch.nn.Parameter(
                param.detach().clone(), requires_grad=param.requires_grad
            )
            cloned_params += 1
        for name, buffer in list(module._buffers.items()):
            if buffer is None:
                continue
            module._buffers[name] = buffer.detach().clone()
            cloned_buffers += 1
    print(f"re-materialised {cloned_params} parameters, {cloned_buffers} buffers")
except Exception:
    traceback.print_exc()
    raise SystemExit("VERDICT: NOT TRAINABLE -- construction failed")

inner = model
total = sum(p.numel() for p in inner.parameters())
trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
print(f"params {total / 1e9:.3f}B | requires_grad {trainable / 1e9:.3f}B")
still_inference = [n for n, p in inner.named_parameters() if p.is_inference()]
print("parameters still inference after cloning:", len(still_inference))
print("is_inference on first param:", next(inner.parameters()).is_inference())

head("forward with a hook that does not detach")
captured = {}


def hook(_module, _inputs, output):
    captured["hidden"] = output[0] if isinstance(output, tuple) else output


target = dict(inner.named_modules()).get(LAYER)
if target is None:
    raise SystemExit(f"VERDICT: layer {LAYER} does not resolve")
handle = target.register_forward_hook(hook)

tokenizer = CharLevelTokenizer(512)
sequences = ["ACGT" * 32, "TTGA" * 32, "GGCA" * 32, "CTAG" * 32]
input_ids = torch.tensor(
    [tokenizer.tokenize(s) for s in sequences], dtype=torch.int
).cuda()
print("input_ids", tuple(input_ids.shape))

torch.cuda.reset_peak_memory_stats()
try:
    with torch.enable_grad():
        model(input_ids)
    hidden = captured.get("hidden")
    print("hidden", tuple(hidden.shape), hidden.dtype, "| requires_grad", hidden.requires_grad)
    print(f"peak after forward {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
except Exception:
    traceback.print_exc()
    handle.remove()
    raise SystemExit("VERDICT: NOT TRAINABLE -- forward failed")
handle.remove()

head("backward through the operator objective")
if hidden is None or not hidden.requires_grad:
    print("VERDICT: NOT TRAINABLE -- no grad on hidden states")
    raise SystemExit(0)

try:
    pooled = hidden.float().mean(dim=1)
    # The matmul cdist path cancels catastrophically above ~25 rows.
    distance = torch.cdist(pooled, pooled, compute_mode="donot_use_mm_for_euclid_dist")
    off = ~torch.eye(len(pooled), dtype=torch.bool, device=distance.device)
    sigma = distance[off].median().clamp_min(1e-6)
    kernel = torch.exp(-(distance * distance) / (2 * sigma * sigma))
    operator = kernel / kernel.sum(-1, keepdim=True)   # diagonal retained
    loss = operator.square().sum()
    loss.backward()

    got = [
        (name, float(p.grad.norm()))
        for name, p in inner.named_parameters()
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
    ]
    print(f"loss {float(loss):.6f} | diagonal min {float(operator.diagonal().min()):.3e}")
    print(f"params with finite non-zero grad: {len(got)} / {sum(1 for _ in inner.parameters())}")
    for name, norm in got[:5]:
        print(f"   {name}  |grad|={norm:.3e}")
    print(f"peak after backward {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("VERDICT: TRAINABLE" if got else "VERDICT: NOT TRAINABLE -- no parameter grads")
except Exception:
    traceback.print_exc()
    print("VERDICT: BACKWARD FAILED")
