"""VariantTableDataModule: row indices as data, labels as metadata, train/test splits."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch  # noqa: E402

from manylatents.dogma.data import VariantTableDataModule  # noqa: E402


def _table():
    return pd.DataFrame({
        "chrom": ["1", "1", "9", "2", "chr21", "9"],
        "pos": [10, 20, 30, 40, 50, 60],
        "ref": list("ACGTAC"), "alt": list("CGTACG"),
        "label": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        "split": ["train", "train", "test", "train", "test", "val"],
        "kind": ["a", "b", "a", "b", "a", "b"],
    })


def _unroll(loader, field="data"):
    return torch.cat([batch[field] for batch in loader])


@pytest.fixture
def path(tmp_path):
    out = tmp_path / "variants.tsv"
    _table().to_csv(out, sep="\t", index=False)
    return out


def test_full_mode_yields_every_row_index(path):
    dm = VariantTableDataModule(path, label_column="label", batch_size=4)
    dm.setup()
    assert dm.mode == "full" and dm.train_dataset is dm.test_dataset
    x = _unroll(dm.train_dataloader())
    assert x.shape == (6, 1) and x.dtype == torch.float32
    np.testing.assert_array_equal(x[:, 0].numpy(), np.arange(6))
    np.testing.assert_array_equal(_unroll(dm.test_dataloader(), "metadata").numpy(),
                                  _table()["label"].to_numpy())
    variants = dm.get_variants()
    assert list(variants["variant_id"][:2]) == ["chr1:10:A>C", "chr1:20:C>G"]
    assert {"label", "split", "kind"} <= set(variants.columns)


def test_split_column(path):
    dm = VariantTableDataModule(path, label_column="label", split_column="split")
    dm.setup()
    assert dm.mode == "split"
    np.testing.assert_array_equal(_unroll(dm.train_dataloader())[:, 0].numpy(), [0, 1, 3])
    np.testing.assert_array_equal(_unroll(dm.test_dataloader())[:, 0].numpy(), [2, 4])
    np.testing.assert_array_equal(dm.train_dataset.get_labels(), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(dm.get_labels(), [0.0, 1.0])
    assert len(dm.get_variants()) == 6  # the "val" row is in neither split


def test_chromosome_split_and_text_labels(tmp_path):
    out = tmp_path / "variants.parquet"
    _table().to_parquet(out)
    dm = VariantTableDataModule(out, label_column="kind", test_chromosomes=[9, "chr21"])
    dm.setup()
    np.testing.assert_array_equal(dm.test_dataset.rows, [2, 4, 5])
    np.testing.assert_array_equal(dm.train_dataset.rows, [0, 1, 3])
    assert dm.label_classes == ["a", "b"]
    np.testing.assert_array_equal(dm.train_dataset.get_labels(), [0, 1, 1])


def test_bad_tables(tmp_path, path):
    with pytest.raises(ValueError, match="not both"):
        VariantTableDataModule(path, split_column="split", test_chromosomes=["9"])
    with pytest.raises(ValueError, match="no column"):
        VariantTableDataModule(path, label_column="score").setup()
    with pytest.raises(ValueError, match="test rows"):
        VariantTableDataModule(path, test_chromosomes=["X"]).setup()
    bare = tmp_path / "bare.tsv"
    _table().drop(columns="alt").to_csv(bare, sep="\t", index=False)
    with pytest.raises(ValueError, match="alt"):
        VariantTableDataModule(bare).setup()


def test_hydra_config_instantiates(path):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    import manylatents.dogma.configs as configs

    cfg = OmegaConf.load(Path(configs.__file__).parent / "data" / "variant_table.yaml")
    cfg.path = str(path)
    dm = instantiate(cfg)
    assert isinstance(dm, VariantTableDataModule) and dm.mode == "full"
