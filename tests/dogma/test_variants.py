"""Canonical variant identifiers shared by the Atlas and TraitGym modules."""
import pandas as pd
import pytest

from manylatents.dogma.variants import variant_id, with_variant_ids


def test_variant_id_adds_chr_prefix_once():
    assert variant_id("1", 1425822, "C", "G") == "chr1:1425822:C>G"
    assert variant_id("chr1", 1425822, "C", "G") == "chr1:1425822:C>G"
    assert variant_id("X", "99", "A", "AT") == "chrX:99:A>AT"


def test_with_variant_ids_adds_columns_and_keeps_input():
    variants = pd.DataFrame(
        {"chrom": ["1", "chrX"], "pos": [10, 20], "ref": ["A", "C"], "alt": ["G", "T"],
         "label": [True, False]}
    )
    out = with_variant_ids(variants)
    assert out["chromosome"].tolist() == ["chr1", "chrX"]
    assert out["variant_id"].tolist() == ["chr1:10:A>G", "chrX:20:C>T"]
    assert out["label"].tolist() == [True, False]
    assert "variant_id" not in variants.columns


def test_with_variant_ids_refuses_missing_columns_and_duplicates():
    with pytest.raises(ValueError, match="alt"):
        with_variant_ids(pd.DataFrame({"chrom": ["1"], "pos": [1], "ref": ["A"]}))
    twice = pd.DataFrame(
        {"chrom": ["1", "1"], "pos": [5, 5], "ref": ["A", "A"], "alt": ["G", "G"]}
    )
    with pytest.raises(ValueError, match="chr1:5:A>G"):
        with_variant_ids(twice)
