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
