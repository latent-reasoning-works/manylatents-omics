"""Assembling the full track matrix from cached scorers."""
import numpy as np
import pandas as pd
import pytest

from manylatents.dogma import atlas
from manylatents.dogma.atlas import AtlasMatrix, AtlasScoresMissing
from manylatents.dogma.benchmarks.track_matrix import load_track_matrix
from manylatents.utils.exceptions import MeasurementUnavailable


def _matrix(ids, n_tracks, offset):
    values = np.array([[offset + 10 * i + t for t in range(n_tracks)] for i in range(len(ids))],
                      dtype=np.float32).reshape(len(ids), n_tracks)
    return AtlasMatrix(values=values, variant_ids=list(ids),
                       tracks=pd.DataFrame({"name": [f"t{t}" for t in range(n_tracks)]}))


@pytest.fixture
def cached(monkeypatch):
    calls = []
    store = {
        "A": _matrix(["v1", "v2", "v3"], 2, 0),
        "B": _matrix(["v3", "v1", "v4"], 1, 100),
        "G": _matrix(["v3"], 2, 1000),
        "H": _matrix(["v5"], 1, -10),
    }

    def fake(cache_dirs, scorer, *, layer=None, gene_reduce=None):
        calls.append((scorer, gene_reduce))
        if scorer not in store:
            raise AtlasScoresMissing(scorer, [])
        return store[scorer]

    monkeypatch.setattr(atlas, "load_cached_scores", fake)
    monkeypatch.setattr(atlas, "cached_variants", lambda directory, scorer: store[scorer].variant_ids if scorer in store else [])
    monkeypatch.setattr(atlas, "missing_variants", lambda directory, scorer: ["v1", "v3"])
    return calls


def test_rows_are_the_variants_every_scorer_has_in_first_scorer_order(cached):
    matrix = load_track_matrix(["dir"], ["A", "B"])
    assert matrix.variant_ids == ["v1", "v3"]
    assert matrix.track_scorer.tolist() == ["A", "A", "B"]
    assert matrix.values.dtype == np.float64
    np.testing.assert_array_equal(matrix.values, [[0, 1, 110], [20, 21, 100]])
    assert matrix.without_rows == {}
    assert matrix.n_dropped == {"A": 1, "B": 1}


def test_gene_scorer_without_a_row_is_zero_filled_and_counted(cached):
    matrix = load_track_matrix(["dir"], ["A", "B"], gene_scorers=["G"])
    assert matrix.variant_ids == ["v1", "v3"]
    assert matrix.track_scorer.tolist() == ["A", "A", "B", "G", "G"]
    np.testing.assert_array_equal(matrix.values[:, 3:], [[0, 0], [1000, 1001]])
    assert matrix.without_rows["G"].tolist() == [True, False]
    assert ("G", "maxabs") in cached and ("A", None) in cached


def test_bad_requests_are_refused(cached):
    with pytest.raises(ValueError, match="at least one"):
        load_track_matrix(["dir"], [])
    with pytest.raises(ValueError, match="both"):
        load_track_matrix(["dir"], ["A"], gene_scorers=["A"])
    with pytest.raises(AtlasScoresMissing):
        load_track_matrix(["dir"], ["A", "Z"])


def test_no_retained_variant_has_a_gene_row(cached):
    matrix = load_track_matrix(["dir"], ["A", "B"], gene_scorers=["H"])
    assert matrix.variant_ids == ["v1", "v3"]
    np.testing.assert_array_equal(matrix.values[:, -1], [0, 0])
    assert matrix.without_rows["H"].tolist() == [True, True]
    assert matrix.n_dropped == {"A": 1, "B": 1, "H": 1}


def test_missing_gene_cache_is_not_assumed_to_mean_no_genes(cached):
    with pytest.raises(AtlasScoresMissing):
        load_track_matrix(["dir"], ["A"], gene_scorers=["Z"])


def test_disjoint_scorers_are_undefined(cached):
    with pytest.raises(MeasurementUnavailable, match="common"):
        load_track_matrix(["dir"], ["A", "H"])


@pytest.mark.parametrize("scorers,genes", [(["A", "A"], []), (["A"], ["G", "G"])])
def test_repeated_scorers_are_refused(cached, scorers, genes):
    with pytest.raises(ValueError, match="duplicate"):
        load_track_matrix(["dir"], scorers, gene_scorers=genes)


def test_real_gene_cache_distinguishes_unattempted_from_no_rows(tmp_path):
    from tests.dogma.test_atlas import FakeAtlasClient, _fetch, _variants
    from manylatents.dogma.variants import with_variant_ids

    variants = _variants(12)
    plain, partial, rest = [tmp_path / name for name in ('plain', 'partial', 'rest')]
    _fetch(FakeAtlasClient(), variants, plain)
    _fetch(FakeAtlasClient(unservable=[1, 3], genes=['g1']), variants.iloc[:4], partial,
           requested_scorers=['G'])
    ids = with_variant_ids(variants)['variant_id'].tolist()
    with pytest.raises(AtlasScoresMissing) as error:
        load_track_matrix([plain, partial], ['S'], gene_scorers=['G'])
    assert set(error.value.variant_ids) == set(ids[4:])
    _fetch(FakeAtlasClient(genes=['g1']), variants.iloc[4:], rest, requested_scorers=['G'])
    matrix = load_track_matrix([plain, partial, rest], ['S'], gene_scorers=['G'])
    expected = np.isin(matrix.variant_ids, [ids[0], ids[2]])
    np.testing.assert_array_equal(matrix.without_rows['G'], expected)
    np.testing.assert_array_equal(matrix.values[expected, 3:], 0)
    assert matrix.n_dropped == {'S': 0, 'G': 0}


def test_real_empty_gene_rows_and_absent_shard_are_distinguished(tmp_path):
    from tests.dogma.test_atlas import FakeAtlasClient, _fetch, _variants
    from manylatents.dogma.variants import with_variant_ids

    class GeneClient(FakeAtlasClient):
        def _anndata(self, variants):
            scored = super()._anndata(variants)
            return scored[[v.position % 2 == 0 for v in scored.obs['variant']]].copy()

    variants = _variants(4)
    plain, gene, absent = [tmp_path / name for name in ('plain', 'gene', 'absent')]
    _fetch(FakeAtlasClient(), variants, plain)
    _fetch(GeneClient(genes=['g1']), variants.iloc[:2], gene, requested_scorers=['G'])
    ids = with_variant_ids(variants)['variant_id'].tolist()
    with pytest.raises(AtlasScoresMissing) as error:
        load_track_matrix([plain, gene, absent], ['S'], gene_scorers=['G'])
    assert set(error.value.variant_ids) == set(ids[2:])
    _fetch(GeneClient(genes=['g1']), variants.iloc[2:], absent, requested_scorers=['G'])
    matrix = load_track_matrix([plain, gene, absent], ['S'], gene_scorers=['G'])
    expected = np.isin(matrix.variant_ids, [ids[0], ids[2]])
    np.testing.assert_array_equal(matrix.without_rows['G'], expected)
    np.testing.assert_array_equal(matrix.values[expected, 3:], 0)
