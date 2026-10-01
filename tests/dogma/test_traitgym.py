"""TraitGym table readers, exercised on local files. No network."""
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from manylatents.dogma.data import traitgym  # noqa: E402

DATASET = "mendelian_traits_matched_9"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A miniature TraitGym repository on disk, served through `_download`."""
    root = tmp_path / DATASET
    (root / "subset").mkdir(parents=True)
    (root / "features").mkdir()
    (root / "preds" / "all").mkdir(parents=True)
    (root / "AUPRC_by_chrom_weighted_average" / "all").mkdir(parents=True)
    variants = pd.DataFrame(
        {"chrom": ["1", "1", "X"], "pos": [10, 20, 30], "ref": ["A", "C", "G"],
         "alt": ["G", "T", "A"], "label": [True, False, False],
         "match_group": ["m0", "m0", "m1"], "consequence": ["PLS", "PLS", "dELS"]}
    )
    variants.to_parquet(root / "test.parquet")
    variants[["chrom", "pos", "ref", "alt"]].to_parquet(root / "subset" / "all.parquet")
    pd.DataFrame({"f0": [0.1, 0.2, 0.3], "f1": [1.0, 2.0, 3.0]}).to_parquet(
        root / "features" / "Toy.parquet"
    )
    pd.DataFrame({"score": [0.9, 0.2, 0.4]}).to_parquet(root / "preds" / "all" / "Toy.parquet")
    pd.DataFrame(
        {"model": ["Toy"], "metric": ["AUPRC"], "score": [0.75], "se": [0.05]}
    ).to_csv(root / "AUPRC_by_chrom_weighted_average" / "all" / "Toy.csv", index=False)

    requested = []

    def fake_download(relative_path, cache_dir=None):
        requested.append(relative_path)
        path = tmp_path / relative_path
        if not path.exists():
            raise FileNotFoundError(relative_path)
        return path

    monkeypatch.setattr(traitgym, "_download", fake_download)
    return requested


def test_load_variants(repo):
    variants = traitgym.load_variants(DATASET)
    assert repo == [f"{DATASET}/test.parquet"]
    assert variants["variant_id"].tolist() == ["chr1:10:A>G", "chr1:20:C>T", "chrX:30:G>A"]
    assert variants["chromosome"].tolist() == ["chr1", "chr1", "chrX"]
    assert variants["label"].dtype == bool
    assert variants["label"].tolist() == [True, False, False]


def test_load_features_is_row_aligned(repo):
    features = traitgym.load_features(DATASET, "Toy")
    assert features.shape == (3, 2)
    assert features["f1"].tolist() == [1.0, 2.0, 3.0]


def test_load_predictions_carries_variant_ids(repo):
    predictions = traitgym.load_predictions(DATASET, "Toy")
    assert predictions["variant_id"].tolist() == ["chr1:10:A>G", "chr1:20:C>T", "chrX:30:G>A"]
    assert predictions["score"].tolist() == [0.9, 0.2, 0.4]


def test_load_published_metric(repo):
    assert traitgym.load_published_metric(DATASET, "Toy") == {"score": 0.75, "se": 0.05}


def test_unknown_dataset_is_refused(repo):
    with pytest.raises(ValueError, match="mendelian_traits_matched_9"):
        traitgym.load_variants("mendelian")


def test_read_variants_refuses_tables_without_labels(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"chrom": ["1"], "pos": [1], "ref": ["A"], "alt": ["G"]}).to_parquet(path)
    with pytest.raises(ValueError, match="label"):
        traitgym.read_variants(path)


def test_misaligned_feature_or_prediction_file_is_refused(repo, tmp_path):
    pd.DataFrame({"f0": [0.1, 0.2]}).to_parquet(
        tmp_path / DATASET / "features" / "Short.parquet"
    )
    with pytest.raises(ValueError, match="rows"):
        traitgym.load_features(DATASET, "Short")
    pd.DataFrame({"score": [0.1, 0.2]}).to_parquet(
        tmp_path / DATASET / "preds" / "all" / "Short.parquet"
    )
    with pytest.raises(ValueError, match="rows"):
        traitgym.load_predictions(DATASET, "Short")


@pytest.mark.parametrize("dataset", traitgym.FULL_DATASETS)
def test_load_pool(repo, tmp_path, dataset):
    root = tmp_path / dataset
    root.mkdir()
    table = pd.read_parquet(tmp_path / DATASET / "test.parquet")
    table.to_parquet(root / "test.parquet")
    loaded = traitgym.load_pool(dataset)
    pd.testing.assert_frame_equal(loaded, traitgym.read_variants(root / "test.parquet"))
    assert repo == [f"{dataset}/test.parquet"]
    with pytest.raises(ValueError, match="full TraitGym"):
        traitgym.load_pool(DATASET)


@pytest.fixture
def pool():
    return pd.DataFrame({
        "variant_id": [f"v{i}" for i in range(24)],
        "label": [False] * 23 + [True],
        "ref": ["A"] * 22 + ["AA", "A"],
        "alt": ["G"] * 21 + ["GG", "G", "G"],
        "consequence": ["a"] * 10 + ["b"] * 14,
    })


def test_uniform_background(pool):
    import numpy as np

    def sample():
        return traitgym.sample_background(pool, 12, np.random.default_rng(7), exclude=["v0"])

    result = sample()
    pd.testing.assert_frame_equal(result, sample())
    assert len(result) == 12
    assert not set(result.variant_id) & {"v0", "v21", "v22", "v23"}
    assert result.index.is_monotonic_increasing
    assert result.attrs["shortfall"] == 0
    scarce = traitgym.sample_background(pool, 30, np.random.default_rng(7))
    assert scarce.attrs["shortfall"] == 9


def test_matched_background(pool):
    import numpy as np

    target = pd.DataFrame({"consequence": ["a", "b", "b", "b"]})
    result = traitgym.sample_background(pool, 8, np.random.default_rng(3), match_to=target)
    assert result.consequence.value_counts().to_dict() == {"b": 6, "a": 2}
    assert result.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(result, traitgym.sample_background(
        pool, 8, np.random.default_rng(3), match_to=target))
    scarce = traitgym.sample_background(pool, 20, np.random.default_rng(3), match_to=target)
    assert scarce.attrs == {"shortfall": 4, "shortfall_by_stratum": {"b": 4}}
    rounded = traitgym.sample_background(pool, 3, np.random.default_rng(3), match_to=target)
    assert len(rounded) == 3
    missing = traitgym.sample_background(
        pool, 2, np.random.default_rng(3), match_to=pd.DataFrame({"consequence": [None]}))
    assert missing.attrs["shortfall"] == 2
