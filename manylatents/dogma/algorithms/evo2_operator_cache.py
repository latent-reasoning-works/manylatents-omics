"""The teacher operator cache.

A teacher is frozen, encoded once, and stored as pooled representations plus
ONE bandwidth (`teacher_sigma`) and the diffusion operator built from it
(`soft_diffop`). Students match that operator; they never derive their own
sigma. The cache on disk is a directory holding `representations.pt`,
`splits.json`, `manifest.json` and `completed.json` -- the manifest records
`pooling`, `layer`, `sigma`, `window_bp` and `model_name`, and a cache missing
the `pooling` marker is refused rather than silently consumed as unpooled.
`pair_and_window_sequences` keeps a variant's ID attached to its sequence
through filtering and windowing, so a missing sequence cannot shift every
later ID out of alignment with the representation it names.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor


def teacher_sigma(representations: Tensor, quantile: float = 0.25) -> Tensor:
    """One bandwidth, taken from the teacher's own pairwise distances.

    `compute_mode="donot_use_mm_for_euclid_dist"` matters: the matmul path
    (`||a||^2 + ||b||^2 - 2 a.b`) cancels catastrophically once rows number
    more than about 25, see `soft_diffop`.
    """
    representations = representations.float()
    distances = torch.cdist(
        representations,
        representations,
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    n = distances.shape[0]
    off_diagonal = distances[~torch.eye(n, dtype=torch.bool, device=distances.device)]
    return torch.quantile(off_diagonal, quantile)


def pair_and_window_sequences(
    variant_ids: List[str],
    sequences: List[str],
    window_bp: int,
) -> Tuple[List[str], List[str], List[Tuple[str, int]]]:
    """Pair each variant ID with its DNA sequence and center it on window_bp.

    IDs and sequences must be filtered together: dropping a missing
    sequence independently of its ID shifts every later ID out of
    alignment with the representation it is meant to describe. A sequence
    shorter than window_bp cannot supply that much context, so it is
    reported in the third return value and excluded rather than silently
    measured at a smaller-than-requested window.

    Returns (kept_variant_ids, windowed_sequences, insufficient), where
    insufficient is a list of (variant_id, actual_length) pairs for
    sequences too short for the requested window.
    """
    kept_ids: List[str] = []
    windowed: List[str] = []
    insufficient: List[Tuple[str, int]] = []
    for variant_id, seq in zip(variant_ids, sequences):
        if not seq:
            continue
        if len(seq) < window_bp:
            insufficient.append((variant_id, len(seq)))
            continue
        mid = len(seq) // 2
        half = window_bp // 2
        kept_ids.append(variant_id)
        windowed.append(seq[mid - half : mid - half + window_bp])
    return kept_ids, windowed, insufficient


def soft_diffop(representations: Tensor, sigma: Tensor) -> Tensor:
    """The Gaussian diffusion operator P_ij = K(x_i, x_j) / sum_j K(x_i, x_j).

    Always computed in float32 -- pooled teacher activations arrive in
    bfloat16, and the operator is exp(-d^2 / 2 sigma^2) over a cdist of
    those vectors, so evaluating the kernel in bf16 erases the near-neighbour
    structure it is built from. The diagonal is retained deliberately: zero
    it and every two-element subset operator becomes [[0, 1], [1, 0]]
    regardless of the data.
    """
    representations = representations.float()
    sigma = sigma.float()
    distances = torch.cdist(
        representations,
        representations,
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    affinity = torch.exp(-(distances**2) / (2 * sigma**2))
    return affinity / affinity.sum(dim=-1, keepdim=True)


@dataclass
class TeacherCache:
    """A loaded teacher operator cache."""

    representations: Tensor
    sigma: Tensor
    pooling: str
    layer: str
    model_name: str
    window_bp: int
    splits: dict


def _default_splits(
    n: int, seed: int = 0, fractions: tuple = (0.7, 0.15, 0.15)
) -> dict:
    """A deterministic train/val/test partition of row indices."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator).tolist()
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def write_teacher_cache(
    cache_dir: Union[str, Path],
    representations: Tensor,
    sigma: Tensor,
    layer: str,
    model_name: str,
    window_bp: int,
    pooling: str,
    variant_ids: Optional[List[str]] = None,
    splits: Optional[dict] = None,
    seed: int = 0,
) -> None:
    """Write a teacher's pooled representations and one derived sigma to disk.

    Publishes an immutable directory atomically; reruns need a fresh path.
    Refuses a non-scalar sigma: a per-tensor sigma rescales with the
    student and the loss can no longer see a global rescaling at all.
    """
    if sigma.numel() != 1:
        raise ValueError(f"sigma must be a scalar, got shape {tuple(sigma.shape)}")

    cache_dir = Path(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"cache is immutable; choose a new directory: {cache_dir}")
    # Same-filesystem rename publishes all four files together. Existing nonempty
    # directories cannot be replaced, including by concurrent writers.
    with tempfile.TemporaryDirectory(prefix=f".{cache_dir.name}-", dir=cache_dir.parent) as tmp:
        staging = Path(tmp) / "cache"
        staging.mkdir()
        torch.save(representations, staging / "representations.pt")

        n = representations.shape[0]
        resolved_splits = splits if splits is not None else _default_splits(n, seed=seed)
        splits_payload = {"variant_ids": variant_ids, **resolved_splits}
        (staging / "splits.json").write_text(json.dumps(splits_payload))

        manifest = {
            "pooling": pooling,
            "layer": layer,
            "sigma": float(sigma),
            "window_bp": window_bp,
            "model_name": model_name,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest))

        (staging / "completed.json").write_text(json.dumps({"num_rows": n}))
        staging.rename(cache_dir)


def load_teacher_cache(cache_dir: Union[str, Path]) -> TeacherCache:
    """Load a teacher operator cache, refusing one missing its markers.

    A cache without a `pooling` marker cannot be told apart from one that was
    never pooled, so it is refused here rather than silently consumed by a
    student that assumes masked-mean rows.
    """
    cache_dir = Path(cache_dir)
    if not (cache_dir / "manifest.json").is_file():
        raise ValueError("incomplete cache: missing manifest.json")
    manifest = json.loads((cache_dir / "manifest.json").read_text())

    if "pooling" not in manifest:
        raise ValueError(
            f"cache at {cache_dir} has no 'pooling' marker in its manifest; "
            "refusing to load an unpooled cache"
        )
    if "sigma" not in manifest:
        raise ValueError(
            f"cache at {cache_dir} has no 'sigma' in its manifest; "
            "refusing to load a cache with no teacher-derived bandwidth"
        )

    for filename in ("completed.json", "splits.json", "representations.pt"):
        if not (cache_dir / filename).is_file():
            raise ValueError(f"incomplete cache: missing {filename}")
    completed = json.loads((cache_dir / "completed.json").read_text())
    representations = torch.load(cache_dir / "representations.pt", map_location="cpu", weights_only=True)
    if completed.get("num_rows") != representations.shape[0]:
        raise ValueError("completed.json num_rows disagrees with representations")
    splits = json.loads((cache_dir / "splits.json").read_text())
    if not all(key in splits for key in ("train", "val", "test", "variant_ids")):
        raise ValueError("splits.json is missing required row mapping fields")
    ids = splits["variant_ids"]
    if ids is not None and len(ids) != representations.shape[0]:
        raise ValueError("splits.json variant_ids disagrees with num_rows")

    return TeacherCache(
        representations=representations,
        sigma=torch.tensor(manifest["sigma"]),
        pooling=manifest["pooling"],
        layer=manifest.get("layer"),
        model_name=manifest.get("model_name"),
        window_bp=manifest.get("window_bp"),
        splits=splits,
    )


def measure_teacher(encoder, sequences, batch_size):
    """Measure frozen inference, reporting warmup and measured OOMs alike.

    Returns (representations or None, JSON-compatible record). The caller must
    persist the record before exiting nonzero on terminal OOM.
    """
    if batch_size < 1 or not sequences:
        raise ValueError("measurement needs sequences and a positive batch_size")
    record = {"status": "completed", "actual_batch_sizes": [],
              "num_oom_events": 0, "oom_phases": [], "num_batches": 0,
              "wall_clock_s_per_batch": None}
    current_bs = min(batch_size, len(sequences))
    chunks = []
    started = None
    torch.cuda.reset_peak_memory_stats()

    def extract(chunk):
        with torch.no_grad():
            result = encoder._extract_embeddings(encoder._tokenize_batch(chunk))
            torch.cuda.synchronize()
            return result.float().cpu()

    for phase in ("warmup", "measurement"):
        cursor = 0
        if phase == "measurement":
            started = time.perf_counter()
        while cursor < len(sequences):
            failed = False
            try:
                result = extract(sequences[cursor:cursor + current_bs])
            except torch.cuda.OutOfMemoryError:
                failed = True
            # Leave the exception scope before clearing CUDA memory, releasing
            # traceback-held activations before retrying.
            if failed:
                record["num_oom_events"] += 1
                record["oom_phases"].append(phase)
                torch.cuda.empty_cache()
                if current_bs == 1:
                    record["status"] = "oom"
                    record["failure_phase"] = phase
                    break
                current_bs = max(1, current_bs // 2)
                continue
            if phase == "warmup":
                del result
                break
            chunks.append(result)
            record["actual_batch_sizes"].append(len(result))
            cursor += len(result)
        if record["status"] == "oom":
            break
    record["num_batches"] = len(chunks)
    record["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    if started is not None and chunks:
        record["wall_clock_s_per_batch"] = (time.perf_counter() - started) / len(chunks)
    return (torch.cat(chunks) if record["status"] == "completed" else None), record
