"""The teacher operator cache.

A teacher is frozen, encoded once, and stored as pooled representations plus
ONE bandwidth (`teacher_sigma`) and the diffusion operator built from it
(`soft_diffop`). Students match that operator; they never derive their own
sigma. The cache on disk is a directory holding `representations.pt`,
`splits.json`, `manifest.json` and `completed.json` -- the manifest records
`pooling`, `layer`, `sigma`, `window_bp` and `model_name`, and a cache missing
the `pooling` marker is refused rather than silently consumed as unpooled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

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

    Refuses a non-scalar sigma: a per-tensor sigma rescales with the
    student and the loss can no longer see a global rescaling at all.
    """
    if sigma.numel() != 1:
        raise ValueError(f"sigma must be a scalar, got shape {tuple(sigma.shape)}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch.save(representations, cache_dir / "representations.pt")

    n = representations.shape[0]
    resolved_splits = splits if splits is not None else _default_splits(n, seed=seed)
    splits_payload = {"variant_ids": variant_ids, **resolved_splits}
    (cache_dir / "splits.json").write_text(json.dumps(splits_payload))

    manifest = {
        "pooling": pooling,
        "layer": layer,
        "sigma": float(sigma),
        "window_bp": window_bp,
        "model_name": model_name,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))

    (cache_dir / "completed.json").write_text(json.dumps({"num_rows": n}))


def load_teacher_cache(cache_dir: Union[str, Path]) -> TeacherCache:
    """Load a teacher operator cache, refusing one missing its markers.

    A cache without a `pooling` marker cannot be told apart from one that was
    never pooled, so it is refused here rather than silently consumed by a
    student that assumes masked-mean rows.
    """
    cache_dir = Path(cache_dir)
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

    representations = torch.load(cache_dir / "representations.pt")

    splits_path = cache_dir / "splits.json"
    splits = json.loads(splits_path.read_text()) if splits_path.is_file() else {}

    return TeacherCache(
        representations=representations,
        sigma=torch.tensor(manifest["sigma"]),
        pooling=manifest["pooling"],
        layer=manifest.get("layer"),
        model_name=manifest.get("model_name"),
        window_bp=manifest.get("window_bp"),
        splits=splits,
    )
