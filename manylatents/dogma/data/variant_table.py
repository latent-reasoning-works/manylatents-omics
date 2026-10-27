"""DataModule over a table of SNVs (chrom, pos, ref, alt), for variant-level encoders.

Encoders such as :class:`~manylatents.dogma.alphagenome.encoder.AlphaGenomeSiteEncoder`
and :class:`~manylatents.dogma.alphagenome.finetune.AlphaGenomeFinetune` read
their sequences from a reference genome, not from the batch. Each dataset here
therefore yields the variant's ROW INDEX in the table as ``data`` (a float32
tensor of shape ``[1]``), and the label (if any) as ``metadata``. The modules map
``x`` back to rows of :meth:`VariantTableDataModule.get_variants`, so ``fit``
sees only the training rows and ``transform`` returns one row per index of
``x``, in the order of ``x``.

Usage:
    >>> dm = VariantTableDataModule("variants.tsv", label_column="label",
    ...                             test_chromosomes=["9", "21"])
    >>> dm.setup()
    >>> dm.get_variants()          # every row: chrom, pos, ref, alt, ...
    >>> dm.train_dataset.data      # [N_train, 1] row indices

Hydra config:
    python -m manylatents.main data=variant_table data.path=variants.tsv \\
        algorithms/latent=alphagenome_site
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from manylatents.dogma.variants import VARIANT_COLUMNS, with_variant_ids

# Row indices travel as float32 (the dtype the rest of the pipeline expects);
# integers are exact in float32 up to 2**24.
MAX_ROWS = 2 ** 24


def read_variant_table(path: Union[str, Path]) -> pd.DataFrame:
    """A ``.parquet``, ``.csv`` or tab-separated table, with chrom, ref and alt as strings."""
    path = Path(path)
    if path.suffix == ".parquet":
        table = pd.read_parquet(path)
    else:
        table = pd.read_csv(path, sep="," if path.suffix == ".csv" else "\t",
                            dtype={"chrom": str, "ref": str, "alt": str}, low_memory=False)
    for column in ("chrom", "ref", "alt"):
        if column in table.columns:
            table[column] = table[column].astype(str)
    return table.reset_index(drop=True)


class VariantRowDataset(Dataset):
    """Rows of a variant table: ``data`` is the row index (``[1]`` float32),
    ``metadata`` the label (``nan`` without a label column)."""

    def __init__(self, rows: np.ndarray, labels: Optional[np.ndarray] = None):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels)
        # experiment.py and the metrics read ``dataset.data``.
        self.data = torch.as_tensor(self.rows, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        label = np.nan if self.labels is None else self.labels[i]
        return {"data": self.data[i],
                "metadata": torch.as_tensor(label, dtype=torch.float32)}

    def get_labels(self) -> Optional[np.ndarray]:
        return self.labels


class VariantTableDataModule(LightningDataModule):
    """A variant table split into training and test rows.

    Args:
        path: ``.tsv`` (default separator), ``.csv`` or ``.parquet`` with
            columns chrom, pos (1-based), ref, alt; other columns are kept.
        label_column: column used as ``y`` (labels of the datasets); none if None.
            Text labels become integer codes into ``label_classes`` (sorted).
        split_column: column naming each row's split; rows whose value is in
            ``train_values`` train, rows in ``test_values`` are the test set,
            and any other row is in neither.
        test_chromosomes: alternatively, rows on these chromosomes (``9`` or
            ``chr9``) are the test set and all other rows train.
        batch_size: rows per batch of the dataloaders.

    Without ``split_column`` or ``test_chromosomes`` the mode is ``full``:
    the training and test sets are the same object, every row. Otherwise the
    mode is ``split``.

    A label used to evaluate a model should not also train it: with a
    supervised encoder, train on the training rows only, or on targets that
    no evaluation uses.
    """

    def __init__(self, path: Union[str, Path], label_column: Optional[str] = None,
                 split_column: Optional[str] = None,
                 train_values: Sequence[str] = ("train",),
                 test_values: Sequence[str] = ("test",),
                 test_chromosomes: Optional[Sequence[Union[str, int]]] = None,
                 batch_size: int = 256):
        super().__init__()
        if split_column is not None and test_chromosomes is not None:
            raise ValueError("pass split_column or test_chromosomes, not both")
        self.save_hyperparameters()
        self.path = Path(path)
        self.label_column = label_column
        self.split_column = split_column
        self.train_values = [str(v) for v in train_values]
        self.test_values = [str(v) for v in test_values]
        self.test_chromosomes = (None if test_chromosomes is None
                                 else [str(c) for c in test_chromosomes])
        self.batch_size = int(batch_size)
        self.mode = "full" if split_column is None and test_chromosomes is None else "split"
        self.label_classes: Optional[list] = None
        self.table: Optional[pd.DataFrame] = None
        self.train_dataset = None
        self.test_dataset = None

    def _labels(self, rows: np.ndarray) -> Optional[np.ndarray]:
        if self.label_column is None:
            return None
        column = self.table[self.label_column]
        if not pd.api.types.is_numeric_dtype(column):
            # text labels become integer codes, in sorted order of label_classes
            self.label_classes = sorted(column.astype(str).unique())
            column = column.astype(str).map({c: i for i, c in enumerate(self.label_classes)})
        return column.to_numpy()[rows]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.table is not None:
            return
        table = read_variant_table(self.path)
        missing = [c for c in VARIANT_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"{self.path} lacks columns {missing}")
        for column in (self.label_column, self.split_column):
            if column is not None and column not in table.columns:
                raise ValueError(f"{self.path} has no column {column!r}")
        if len(table) >= MAX_ROWS:
            raise ValueError(f"{len(table)} rows: row indices are float32, exact below {MAX_ROWS}")
        self.table = with_variant_ids(table)
        every = np.arange(len(self.table))
        if self.mode == "full":
            dataset = VariantRowDataset(every, self._labels(every))
            self.train_dataset = self.test_dataset = dataset
            return
        if self.split_column is not None:
            split = self.table[self.split_column].astype(str).to_numpy()
            train = every[np.isin(split, self.train_values)]
            test = every[np.isin(split, self.test_values)]
        else:
            wanted = {c if c.startswith("chr") else f"chr{c}" for c in self.test_chromosomes}
            on_test = self.table["chromosome"].isin(wanted).to_numpy()
            train, test = every[~on_test], every[on_test]
        if not len(train) or not len(test):
            raise ValueError(f"split leaves {len(train)} training and {len(test)} test rows")
        self.train_dataset = VariantRowDataset(train, self._labels(train))
        self.test_dataset = VariantRowDataset(test, self._labels(test))

    def get_variants(self) -> pd.DataFrame:
        """Every row of the table (training and test), in file order: chrom, pos,
        ref, alt, ``chromosome``, ``variant_id`` and the other columns. Row ``i``
        is the variant whose index ``i`` the datasets yield."""
        if self.table is None:
            self.setup()
        return self.table

    def get_labels(self) -> Optional[np.ndarray]:
        """Labels of the test rows (all rows in ``full`` mode)."""
        if self.test_dataset is None:
            self.setup()
        return self.test_dataset.get_labels()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def __repr__(self) -> str:
        n = "not set up" if self.table is None else f"{len(self.table)} rows"
        return f"VariantTableDataModule(path={str(self.path)!r}, mode={self.mode!r}, {n})"
