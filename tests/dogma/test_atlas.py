"""Atlas fetch cache and matrix reader, exercised with a fake client. No network."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from manylatents.dogma.atlas import (  # noqa: E402
    AtlasScoresMissing,
    fetch_atlas_scores,
    load_atlas_matrix,
    missing_variants,
)
from manylatents.dogma.variants import with_variant_ids  # noqa: E402

TRACKS = pd.DataFrame(
    {"name": ["t0", "t1", "t2"], "ontology_curie": ["CL:1", "CL:2", None]},
    index=["0", "1", "2"],
)


@dataclass(frozen=True)
class FakeVariant:
    chromosome: str
    position: int
    reference_bases: str
    alternate_bases: str
    name: str = ""


def _score(variant: FakeVariant, track: int) -> float:
    return float(variant.position * 10 + track)


class FakeAtlasClient:
    """Mimics AtlasClient: rows in reversed order, unservable variants raise ValueError."""

    def __init__(self, unservable=(), genes=None, with_quantiles=True):
        self.unservable = set(unservable)
        self.genes = genes
        self.with_quantiles = with_quantiles
        self.batch_calls = 0
        self.single_calls = 0

    def _anndata(self, variants):
        rows, obs = [], []
        for variant in reversed(list(variants)):
            for gene in self.genes or [None]:
                offset = 0.0 if gene is None else {"g1": 0.0, "g2": -1000.0}[gene]
                rows.append([_score(variant, t) + offset for t in range(len(TRACKS))])
                row = {"variant": variant}
                if gene is not None:
                    row["gene_id"] = gene
                obs.append(row)
        x = np.asarray(rows, dtype=np.float32)
        layers = {"quantiles": x / 1e6} if self.with_quantiles else None
        return anndata.AnnData(
            X=x,
            obs=pd.DataFrame(obs, index=[str(i) for i in range(len(obs))]),
            var=TRACKS.copy(),
            layers=layers,
        )

    def query_variants(self, variants, *, requested_scorers, ontology_terms=None,
                       progress_bar=True, max_workers=10, **_):
        self.batch_calls += 1
        if any(v.position in self.unservable for v in variants):
            raise ValueError("variant not found")
        return {name: self._anndata(variants) for name in requested_scorers}

    def query_variant(self, variant, *, requested_scorers, ontology_terms=None, **_):
        self.single_calls += 1
        if variant.position in self.unservable:
            raise ValueError("variant not found")
        return {name: self._anndata([variant]) for name in requested_scorers}


def _variants(n):
    return pd.DataFrame(
        {"chrom": ["1"] * n, "pos": list(range(1, n + 1)), "ref": ["A"] * n, "alt": ["G"] * n}
    )


def _factory(row):
    return FakeVariant(row.chromosome, int(row.pos), row.ref, row.alt, row.variant_id)


def _fetch(client, variants, cache_dir, **kwargs):
    kwargs.setdefault("requested_scorers", ["S"])
    kwargs.setdefault("chunk_size", 4)
    return fetch_atlas_scores(client, variants, cache_dir=cache_dir,
                              variant_factory=_factory, **kwargs)


def test_matrix_is_aligned_to_the_requested_order(tmp_path):
    variants = _variants(10)
    _fetch(FakeAtlasClient(), variants, tmp_path)
    ids = with_variant_ids(variants)["variant_id"].tolist()
    wanted = [ids[7], ids[0], ids[3]]
    matrix = load_atlas_matrix(tmp_path, "S", wanted)
    assert matrix.variant_ids == wanted
    assert matrix.values.dtype == np.float32
    np.testing.assert_array_equal(
        matrix.values, [[80, 81, 82], [10, 11, 12], [40, 41, 42]]
    )
    assert matrix.tracks["name"].tolist() == ["t0", "t1", "t2"]
    assert len(matrix.tracks) == matrix.values.shape[1]


def test_quantiles_layer(tmp_path):
    variants = _variants(5)
    _fetch(FakeAtlasClient(), variants, tmp_path)
    ids = with_variant_ids(variants)["variant_id"].tolist()
    raw = load_atlas_matrix(tmp_path, "S", ids)
    quantiles = load_atlas_matrix(tmp_path, "S", ids, layer="quantiles")
    np.testing.assert_allclose(quantiles.values, raw.values / 1e6, rtol=1e-6)


def test_absent_layer_is_refused(tmp_path):
    variants = _variants(3)
    _fetch(FakeAtlasClient(with_quantiles=False), variants, tmp_path)
    ids = with_variant_ids(variants)["variant_id"].tolist()
    with pytest.raises(KeyError, match="quantiles"):
        load_atlas_matrix(tmp_path, "S", ids, layer="quantiles")


def test_resume_makes_no_request_for_finished_chunks(tmp_path):
    variants = _variants(10)
    first = FakeAtlasClient()
    manifest = _fetch(first, variants, tmp_path)
    assert first.batch_calls == 3                       # chunks of 4, 4, 2
    assert all(c["status"] == "done" for c in manifest["chunks"].values())
    second = FakeAtlasClient()
    _fetch(second, variants, tmp_path)
    assert second.batch_calls == 0 and second.single_calls == 0


def test_unservable_variant_is_isolated_and_reported(tmp_path):
    variants = _variants(10)
    client = FakeAtlasClient(unservable={6})
    _fetch(client, variants, tmp_path)
    ids = with_variant_ids(variants)["variant_id"].tolist()
    assert missing_variants(tmp_path, "S") == [ids[5]]
    present = [i for i in ids if i != ids[5]]
    matrix = load_atlas_matrix(tmp_path, "S", present)
    assert matrix.values.shape == (9, 3)
    with pytest.raises(AtlasScoresMissing) as err:
        load_atlas_matrix(tmp_path, "S", ids)
    assert err.value.variant_ids == [ids[5]]
    assert np.isfinite(matrix.values).all()


def test_other_errors_propagate_and_leave_finished_chunks_usable(tmp_path):
    class Flaky(FakeAtlasClient):
        def query_variants(self, variants, **kwargs):
            if any(v.position == 6 for v in variants):
                raise PermissionError("bad key")
            return super().query_variants(variants, **kwargs)

    variants = _variants(10)
    with pytest.raises(PermissionError):
        _fetch(Flaky(), variants, tmp_path)
    ids = with_variant_ids(variants)["variant_id"].tolist()
    assert load_atlas_matrix(tmp_path, "S", ids[:4]).values.shape == (4, 3)
    resumed = FakeAtlasClient()
    _fetch(resumed, variants, tmp_path)
    assert resumed.batch_calls == 2                     # only the two unfinished chunks


def test_changed_request_into_the_same_cache_is_refused(tmp_path):
    variants = _variants(6)
    _fetch(FakeAtlasClient(), variants, tmp_path)
    with pytest.raises(ValueError, match="requested_scorers"):
        _fetch(FakeAtlasClient(), variants, tmp_path, requested_scorers=["S", "T"])
    with pytest.raises(ValueError, match="chunk_size"):
        _fetch(FakeAtlasClient(), variants, tmp_path, chunk_size=2)


def test_changed_variant_list_refetches_that_chunk(tmp_path):
    _fetch(FakeAtlasClient(), _variants(4), tmp_path)
    shifted = _variants(4).assign(pos=[1, 2, 3, 9])
    client = FakeAtlasClient()
    _fetch(client, shifted, tmp_path)
    assert client.batch_calls == 1
    ids = with_variant_ids(shifted)["variant_id"].tolist()
    np.testing.assert_array_equal(
        load_atlas_matrix(tmp_path, "S", ids).values[3], [90, 91, 92]
    )


def test_gene_scoped_scores_need_an_explicit_reduction(tmp_path):
    variants = _variants(3)
    _fetch(FakeAtlasClient(genes=["g1", "g2"]), variants, tmp_path)
    ids = with_variant_ids(variants)["variant_id"].tolist()
    with pytest.raises(ValueError, match="gene_reduce"):
        load_atlas_matrix(tmp_path, "S", ids)
    reduced = load_atlas_matrix(tmp_path, "S", ids, gene_reduce="maxabs")
    # g2 rows are the g1 rows minus 1000, so they have the larger magnitude
    np.testing.assert_array_equal(reduced.values[0], [10 - 1000, 11 - 1000, 12 - 1000])


def test_track_metadata_with_missing_values_round_trips(tmp_path):
    variants = _variants(2)
    _fetch(FakeAtlasClient(), variants, tmp_path)
    ids = with_variant_ids(variants)["variant_id"].tolist()
    tracks = load_atlas_matrix(tmp_path, "S", ids).tracks
    assert tracks["ontology_curie"].tolist()[:2] == ["CL:1", "CL:2"]
    assert pd.isna(tracks["ontology_curie"].tolist()[2]) or tracks["ontology_curie"].tolist()[2] in ("", "None", "nan")


def test_shortened_request_discards_old_chunks(tmp_path):
    _fetch(FakeAtlasClient(), _variants(8), tmp_path)
    _fetch(FakeAtlasClient(), _variants(4), tmp_path)
    with pytest.raises(AtlasScoresMissing):
        load_atlas_matrix(tmp_path, "S", ["chr1:8:A>G"])


def test_interrupted_write_is_not_read_and_is_refetched(tmp_path, monkeypatch):
    original = anndata.AnnData.write_h5ad

    def interrupted(self, filename, *args, **kwargs):
        original(self, filename, *args, **kwargs)
        raise OSError("interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(anndata.AnnData, "write_h5ad", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            _fetch(FakeAtlasClient(), _variants(4), tmp_path)
    with pytest.raises(AtlasScoresMissing):
        load_atlas_matrix(tmp_path, "S", ["chr1:1:A>G"])
    client = FakeAtlasClient()
    _fetch(client, _variants(4), tmp_path)
    assert client.batch_calls == 1
    assert load_atlas_matrix(tmp_path, "S", ["chr1:1:A>G"]).values[0, 0] == 10


def test_track_order_mismatch_is_refused(tmp_path):
    _fetch(FakeAtlasClient(), _variants(8), tmp_path)
    path = tmp_path / "S" / "chunk_00001.h5ad"
    data = anndata.read_h5ad(path)
    data[:, ::-1].copy().write_h5ad(path)
    with pytest.raises(ValueError, match="track"):
        load_atlas_matrix(tmp_path, "S", ["chr1:1:A>G", "chr1:8:A>G"])


def test_all_variants_missing_are_recorded_without_rows(tmp_path):
    _fetch(FakeAtlasClient(unservable={1, 2}), _variants(2), tmp_path)
    ids = ["chr1:1:A>G", "chr1:2:A>G"]
    assert missing_variants(tmp_path, "S") == ids
    with pytest.raises(AtlasScoresMissing) as err:
        load_atlas_matrix(tmp_path, "S", ids)
    assert err.value.variant_ids == ids
    assert not list(tmp_path.glob("S/*.h5ad"))


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5])
def test_invalid_chunk_size_is_refused(tmp_path, chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        _fetch(FakeAtlasClient(), _variants(2), tmp_path, chunk_size=chunk_size)


class _WithMetadata(FakeAtlasClient):
    """A client that, like the real one, can list its scorers."""

    def scorer_metadata(self):
        return {"S": object(), "CenterMask(output=DNASE, width=501)/v1": object()}


def test_unknown_scorer_is_refused_before_any_request(tmp_path):
    client = _WithMetadata()
    with pytest.raises(ValueError, match="DNASE_typo"):
        _fetch(client, _variants(4), tmp_path, requested_scorers=["S", "DNASE_typo"])
    assert client.batch_calls == 0 and client.single_calls == 0
    assert not (tmp_path / "manifest.json").exists()


def test_scorer_names_unsafe_for_paths_are_cached_and_read_back(tmp_path):
    name = "CenterMask(output=DNASE, width=501)/v1"
    variants = _variants(5)
    _fetch(_WithMetadata(), variants, tmp_path, requested_scorers=[name, "S"])
    ids = with_variant_ids(variants)["variant_id"].tolist()
    np.testing.assert_array_equal(
        load_atlas_matrix(tmp_path, name, ids).values,
        load_atlas_matrix(tmp_path, "S", ids).values,
    )
    assert missing_variants(tmp_path, name) == []
    # one directory per scorer, directly under the cache, nothing nested
    directories = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert len(directories) == 2 and "S" in directories
    assert all(p.parent.parent == tmp_path for p in tmp_path.rglob("*.h5ad"))
