"""Chromosome splits."""
import numpy as np
import pytest

from manylatents.dogma.benchmarks.splits import (
    AVI_TRAIN_CHROMOSOMES,
    AVI_TUNE_CHROMOSOMES,
    chromosome_split,
)


def test_avi_split_assigns_every_chromosome_once():
    chroms = [str(i) for i in range(1, 23)] + ["X", "Y"]
    split = chromosome_split(chroms)
    assert dict(zip(chroms, split))["1"] == "train"
    assert dict(zip(chroms, split))["X"] == "tune"
    report = [c for c, s in zip(chroms, split) if s == "report"]
    assert report == ["3", "6", "9", "12", "16", "18", "19", "21", "Y"]
    assert not AVI_TRAIN_CHROMOSOMES & AVI_TUNE_CHROMOSOMES


def test_chr_prefix_and_integer_names_are_accepted():
    split = chromosome_split(np.array(["chr1", 2, "chr3", "3"], dtype=object))
    assert split.tolist() == ["train", "tune", "report", "report"]


def test_custom_split_and_overlap():
    assert chromosome_split(["5", "6"], train={"chr5"}, tune={"6"}).tolist() == ["train", "tune"]
    with pytest.raises(ValueError, match="both"):
        chromosome_split(["1"], train={"1"}, tune={"chr1"})


def test_prefixes_agree_in_every_partition():
    assert chromosome_split([1, "chr1", "X", "chrX", 3, "chr3"]).tolist() == [
        "train", "train", "tune", "tune", "report", "report"]
    assert chromosome_split([]).shape == (0,)


def test_normalizes_numeric_case_and_whitespace_names_and_partitions():
    assert chromosome_split([1.0, ' Chr4 ', 'CHR2', ' chrx ', 'y', 'mt']).tolist() == [
        'train', 'train', 'tune', 'tune', 'report', 'report']
    assert chromosome_split([' chr1 ', 2.0], train=[1.0], tune=[' CHR2 ']).tolist() == ['train', 'tune']


@pytest.mark.parametrize('bad', [None, np.nan, '23', b'1', '1.5', '1_KI270706v1_random'])
@pytest.mark.parametrize('where', ['chroms', 'train', 'tune'])
def test_invalid_chromosomes_fail_closed(bad, where):
    args = dict(chroms=['1'], train=['1'], tune=['2'])
    args[where] = [bad]
    with pytest.raises(ValueError, match='chromosome') as error:
        chromosome_split(**args)
    assert repr(bad) in str(error.value)
