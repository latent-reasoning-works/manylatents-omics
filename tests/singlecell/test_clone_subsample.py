"""clone_subsample: whole-clone subsampling for lineage-traced single-cell data."""

import numpy as np
import pandas as pd
import pytest

from manylatents.singlecell.sampling import clone_subsample


def reference_clone_subsample(obs, n, seed=0):
    """Verbatim copy of the original research implementation (the behavioural reference).

    The one edit: ``keep.append(c); total += size[c]`` is split over two lines.
    """
    day = obs["Time point"].to_numpy()
    clone = obs["clone"].to_numpy()
    later = pd.Series(clone[np.isin(day, [4, 6])]).value_counts()
    early = set(clone[day == 2])
    eligible = np.array(sorted(c for c in early if later.get(c, 0) >= 3))
    rng = np.random.default_rng(seed)
    size = pd.Series(clone).value_counts()
    keep, total = [], 0
    for c in rng.permutation(eligible):
        if total + size[c] > n:
            continue
        keep.append(c)
        total += size[c]
    return np.flatnonzero(np.isin(clone, keep))


def synthetic_obs(n_clones=60, seed=7):
    """Clones of varied size; some lack an early cell, some have too few late cells."""
    rng = np.random.default_rng(seed)
    days, clones = [], []
    for c in range(n_clones):
        n_early = int(rng.integers(0, 3))
        n_late = int(rng.integers(0, 9))
        days += [2] * n_early + list(rng.choice([4, 6], size=n_late))
        clones += [c] * (n_early + n_late)
    order = rng.permutation(len(days))  # interleave clones, as in a real cell table
    return pd.DataFrame({
        "Time point": np.asarray(days)[order],
        "clone": np.asarray(clones)[order],
    })


@pytest.mark.parametrize("seed", [0, 1, 2, 13])
@pytest.mark.parametrize("n", [1, 10, 50, 120, 250, 10_000])
def test_matches_reference(n, seed):
    obs = synthetic_obs()
    got = clone_subsample(obs, n, seed=seed)
    want = reference_clone_subsample(obs, n, seed=seed)
    assert got.dtype == want.dtype
    assert got.tobytes() == want.tobytes()


def test_default_seed_matches_reference():
    obs = synthetic_obs(seed=3)
    assert clone_subsample(obs, 80).tobytes() == reference_clone_subsample(obs, 80).tobytes()


def test_keeps_whole_clones_and_respects_budget():
    obs = synthetic_obs()
    idx = clone_subsample(obs, 120, seed=1)
    assert len(idx) <= 120
    assert np.all(np.diff(idx) > 0)
    chosen = set(obs["clone"].to_numpy()[idx])
    assert set(np.flatnonzero(obs["clone"].isin(chosen))) == set(idx)


def test_n_smaller_than_every_clone_is_empty():
    obs = synthetic_obs()
    smallest = obs["clone"].value_counts().min()
    idx = clone_subsample(obs, smallest - 1, seed=0)
    assert idx.size == 0
    assert idx.tobytes() == reference_clone_subsample(obs, smallest - 1).tobytes()


def test_no_eligible_clones_is_empty():
    obs = pd.DataFrame({"Time point": [2, 4, 4, 6, 4, 6, 6], "clone": [0, 0, 0, 1, 1, 1, 1]})
    # clone 0: early cell but only 2 late cells; clone 1: 4 late cells but no early cell.
    idx = clone_subsample(obs, 100)
    assert idx.size == 0
    assert idx.tobytes() == reference_clone_subsample(obs, 100).tobytes()


def test_n_larger_than_total_keeps_every_eligible_clone():
    obs = synthetic_obs()
    idx = clone_subsample(obs, 10 * len(obs), seed=5)
    day, clone = obs["Time point"].to_numpy(), obs["clone"].to_numpy()
    late = pd.Series(clone[np.isin(day, [4, 6])]).value_counts()
    eligible = {c for c in set(clone[day == 2]) if late.get(c, 0) >= 3}
    assert set(clone[idx]) == eligible
    assert np.array_equal(idx, np.flatnonzero(np.isin(clone, list(eligible))))


def test_custom_columns_and_days():
    obs = synthetic_obs().rename(columns={"Time point": "day", "clone": "lineage"})
    obs["day"] = obs["day"].map({2: 0, 4: 1, 6: 2})
    renamed = obs.rename(columns={"day": "Time point", "lineage": "clone"})
    renamed["Time point"] = renamed["Time point"].map({0: 2, 1: 4, 2: 6})
    got = clone_subsample(obs, 100, seed=2, time_col="day", clone_col="lineage",
                          early=(0,), late=(1, 2))
    assert got.tobytes() == reference_clone_subsample(renamed, 100, seed=2).tobytes()


def test_min_late_threshold():
    obs = pd.DataFrame({"Time point": [2, 4, 6, 2, 4, 4, 6], "clone": [0, 0, 0, 1, 1, 1, 1]})
    assert clone_subsample(obs, 100).tolist() == [3, 4, 5, 6]
    assert clone_subsample(obs, 100, min_late=2).tolist() == list(range(7))
