"""TraitGym benchmark tables (Benegas, Eraslan and Song, 2025).

TraitGym is a pair of non-coding variant benchmarks hosted as a Hugging Face
dataset: causal variants for Mendelian traits and for complex traits, each
with nine matched controls per positive. This module downloads individual
files on demand and reads them with a canonical ``variant_id``.

Layout of the dataset repository::

    {dataset}/test.parquet                    variants, labels, match groups
    {dataset}/subset/{subset}.parquet         variants in a subset
    {dataset}/features/{features}.parquet     one row per variant, row-aligned with test.parquet
    {dataset}/preds/{subset}/{model}.parquet  one `score` column, row-aligned with the subset
    {dataset}/{metric}/{subset}/{model}.csv   model, metric, score, se
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from manylatents.dogma.variants import with_variant_ids

REPO_ID = "songlab/TraitGym"
FULL_DATASETS = ("mendelian_traits_all", "complex_traits_all")
DATASETS = ("mendelian_traits_matched_9", "complex_traits_matched_9")


def _download(relative_path: str, cache_dir: Optional[str] = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(REPO_ID, relative_path, repo_type="dataset", cache_dir=cache_dir)
    )


def _check_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"unknown TraitGym dataset {dataset!r}; choose from {DATASETS}")


def read_variants(path) -> pd.DataFrame:
    """A TraitGym variant table with ``chromosome`` and ``variant_id`` added."""
    table = pd.read_parquet(path)
    if "label" not in table.columns:
        raise ValueError(f"{path} has no label column")
    table = with_variant_ids(table)
    table["label"] = table["label"].astype(bool)
    return table.reset_index(drop=True)


def load_variants(dataset: str, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Variants, labels and match groups of one TraitGym dataset."""
    _check_dataset(dataset)
    return read_variants(_download(f"{dataset}/test.parquet", cache_dir))


def load_features(dataset: str, features: str, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """A feature table, one row per variant in the order of :func:`load_variants`."""
    _check_dataset(dataset)
    table = pd.read_parquet(_download(f"{dataset}/features/{features}.parquet", cache_dir))
    n_variants = len(pd.read_parquet(_download(f"{dataset}/test.parquet", cache_dir),
                                     columns=["pos"]))
    if len(table) != n_variants:
        raise ValueError(
            f"features {features!r} have {len(table)} rows, the dataset has {n_variants}"
        )
    return table.reset_index(drop=True)


def load_predictions(
    dataset: str, model: str, subset: str = "all", cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """A model's published scores for a subset, joined to that subset's variants."""
    _check_dataset(dataset)
    variants = with_variant_ids(
        pd.read_parquet(_download(f"{dataset}/subset/{subset}.parquet", cache_dir))
    ).reset_index(drop=True)
    scores = pd.read_parquet(_download(f"{dataset}/preds/{subset}/{model}.parquet", cache_dir))
    if len(scores) != len(variants):
        raise ValueError(
            f"predictions {model!r} have {len(scores)} rows, subset {subset!r} has {len(variants)}"
        )
    variants["score"] = scores["score"].to_numpy()
    return variants


def load_published_metric(
    dataset: str,
    model: str,
    metric: str = "AUPRC_by_chrom_weighted_average",
    subset: str = "all",
    cache_dir: Optional[str] = None,
) -> dict:
    """The value TraitGym publishes for a model: ``{"score": ..., "se": ...}``."""
    _check_dataset(dataset)
    table = pd.read_csv(_download(f"{dataset}/{metric}/{subset}/{model}.csv", cache_dir))
    if len(table) != 1:
        raise ValueError(f"expected one row in the metric file, found {len(table)}")
    row = table.iloc[0]
    return {"score": float(row["score"]), "se": float(row["se"])}


def load_pool(dataset: str, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Full TraitGym variant table, including labels and canonical identifiers."""
    if dataset not in FULL_DATASETS:
        raise ValueError(f"unknown full TraitGym dataset {dataset!r}; choose from {FULL_DATASETS}")
    return read_variants(_download(f"{dataset}/test.parquet", cache_dir))


def sample_background(pool, n, rng, exclude=(), match_to=None, column="consequence") -> pd.DataFrame:
    """Sample negative SNVs without replacement, returning rows in pool order.

    ``rng`` is a NumPy Generator. Exclude canonical variant IDs with ``exclude``.
    With ``match_to``, allocate ``n`` rows proportionally to its ``column``
    values (including missing values), using largest remainders to round to
    integers; ties follow first appearance in the target. Otherwise sample
    uniformly. An empty matching table is invalid for a nonzero request.

    Scarce strata are not filled from other strata. ``result.attrs["shortfall"]``
    is the total unfilled quota; ``result.attrs["shortfall_by_stratum"]`` maps
    deficient strata to their shortfall (empty for uniform sampling).
    """
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    eligible = ((pool["label"] == False) & (pool["ref"].str.len() == 1) &
                (pool["alt"].str.len() == 1) & ~pool["variant_id"].isin(exclude))
    candidates = np.flatnonzero(eligible.fillna(False).to_numpy(dtype=bool))
    shortfalls = {}
    if match_to is None:
        rows = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    else:
        if n and match_to.empty:
            raise ValueError("match_to must not be empty when n is positive")
        counts = match_to[column].value_counts(sort=False, dropna=False)
        quotas = counts.to_numpy(dtype=float) / max(len(match_to), 1) * n
        wants = np.floor(quotas).astype(int)
        order = np.argsort(-(quotas - wants), kind="stable")
        wants[order[:n - wants.sum()]] += 1
        parts = []
        for value, want in zip(counts.index, wants):
            matches = pool[column].isna() if pd.isna(value) else pool[column].eq(value)
            available = candidates[matches.iloc[candidates].fillna(False).to_numpy(dtype=bool)]
            take = min(int(want), len(available))
            if take < want:
                shortfalls[value] = int(want) - take
            parts.append(rng.choice(available, size=take, replace=False))
        rows = np.concatenate(parts) if parts else np.array([], dtype=int)
    result = pool.iloc[np.sort(rows)].copy()
    result.attrs["shortfall"] = n - len(result)
    result.attrs["shortfall_by_stratum"] = shortfalls
    return result
