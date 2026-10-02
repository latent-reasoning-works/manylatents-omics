"""Per-variant lookups in a tabix-indexed score table, on a tiny file built here."""
import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")

from manylatents.dogma.variants import lookup_tabix_scores  # noqa: E402

ROWS = [
    ("chr1", 100, "A", "C", -0.5, 1.0),
    ("chr1", 100, "A", "G", 0.25, 12.5),
    ("chr1", 100, "A", "T", 0.0, 5.0),
    ("chr1", 205, "G", "A", 1.5, 30.0),
    ("chr2", 50, "T", "C", 0.75, 20.0),
]


def _table(tmp_path, header=True, chr_prefix=True):
    path = tmp_path / "scores.tsv"
    with open(path, "w") as handle:
        if header:
            handle.write("#CHROM\tPOS\tREF\tALT\traw_score\tPHRED\n")
        for chrom, pos, ref, alt, raw, phred in ROWS:
            chrom = chrom if chr_prefix else chrom[3:]
            handle.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{raw}\t{phred}\n")
    pysam.tabix_compress(str(path), str(path) + ".gz", force=True)
    pysam.tabix_index(str(path) + ".gz", seq_col=0, start_col=1, end_col=1, force=True)
    return str(path) + ".gz"


VARIANTS = pd.DataFrame({
    "chrom": ["2", "1", "1", "1", "7"],
    "pos": [50, 100, 205, 100, 10],
    "ref": ["T", "A", "C", "A", "A"],     # third row: reference differs from the table's
    "alt": ["C", "G", "A", "N", "G"],     # fourth row: allele not in the table
})


def test_values_come_back_in_input_order_with_the_tables_column_names(tmp_path):
    out = lookup_tabix_scores(_table(tmp_path), VARIANTS)
    assert out["variant_id"].tolist() == [
        "chr2:50:T>C", "chr1:100:A>G", "chr1:205:C>A", "chr1:100:A>N", "chr7:10:A>G"]
    assert out["found"].tolist() == [True, True, True, False, False]
    assert out["raw_score"].tolist()[:3] == [0.75, 0.25, 1.5]
    assert out["PHRED"].tolist()[:3] == [20.0, 12.5, 30.0]
    assert out["raw_score"].isna().tolist() == [False, False, False, True, True]


def test_reference_disagreement_is_flagged_not_hidden(tmp_path):
    out = lookup_tabix_scores(_table(tmp_path), VARIANTS)
    assert out["reference_matches"].tolist()[:3] == [True, True, False]
    assert out["table_ref"].tolist()[:3] == ["T", "A", "G"]
    assert out["reference_matches"].isna().tolist()[3:] == [True, True]


def test_table_without_chr_prefix_and_without_header(tmp_path):
    out = lookup_tabix_scores(_table(tmp_path, header=False, chr_prefix=False), VARIANTS)
    assert out["found"].tolist() == [True, True, True, False, False]
    assert out["value_0"].tolist()[:3] == [0.75, 0.25, 1.5]
    assert out["value_1"].tolist()[:3] == [20.0, 12.5, 30.0]


def test_missing_columns_are_refused(tmp_path):
    with pytest.raises(ValueError, match="alt"):
        lookup_tabix_scores(_table(tmp_path), VARIANTS.drop(columns="alt"))
