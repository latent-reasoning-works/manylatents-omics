"""ncVarDB reader, exercised on local files. No network."""
import pandas as pd
import pytest

from manylatents.dogma.data import ncvardb

PATHOGENIC = (
    "ID\tgenome\tchr\tpos\tref\talt\tmut_type\tmut_position\tx_ref\tpmed_ID\tphenotype\n"
    "ncVarP_1\thg38\t1\t100\tC\tT\tsubstitution\tintronic\tx\t1\tp\n"
    "ncVarP_2\thg38\tX\t200\tAAGT\t-\tdeletion\t5utr\tx\t1\tp\n"
    "ncVarP_3\thg38\tM\t300\tA\tG\tsubstitution\tncRNA\tx\t1\tp\n"
)
BENIGN = (
    "ID\tgenome\tchr\tpos\tref\talt\tmut_type\tmut_position\tMAF\tx_ref\n"
    "ncVarB_1\thg38\t3\t400\tG\tA\tsubstitution\t3utr\t0.2\trs1\n"
    "ncVarB_2\thg38\t3\t500\t-\tT\tinsertion\tintronic\t0.1\trs2\n"
)


def _write(tmp_path, pathogenic=PATHOGENIC, benign=BENIGN):
    p, b = tmp_path / "p.tsv", tmp_path / "b.tsv"
    p.write_text(pathogenic)
    b.write_text(benign)
    return p, b


def test_snvs_are_kept_with_labels_and_ids(tmp_path):
    table = ncvardb.read_variants(*_write(tmp_path))
    assert table["variant_id"].tolist() == ["chr1:100:C>T", "chr3:400:G>A"]
    assert table["label"].tolist() == [True, False]
    assert table["region"].tolist() == ["intronic", "3utr"]
    assert table["source_id"].tolist() == ["ncVarP_1", "ncVarB_1"]
    assert table["pos"].dtype.kind == "i"
    assert table["chrom"].tolist() == ["1", "3"]


def test_excluded_rows_are_counted(tmp_path):
    table = ncvardb.read_variants(*_write(tmp_path))
    assert table.attrs["n_not_snv"] == 2
    assert table.attrs["n_mitochondrial"] == 1


def test_other_genome_builds_are_refused(tmp_path):
    with pytest.raises(ValueError, match="hg38"):
        ncvardb.read_variants(*_write(tmp_path, benign=BENIGN.replace("hg38", "hg19")))


def test_missing_columns_are_refused(tmp_path):
    with pytest.raises(ValueError, match="mut_position"):
        ncvardb.read_variants(*_write(tmp_path, benign=BENIGN.replace("mut_position", "where")))


def test_load_variants_reads_the_downloaded_files(tmp_path, monkeypatch):
    p, b = _write(tmp_path)
    monkeypatch.setattr(ncvardb, "_download", lambda name, cache_dir=None: {"pathogenic": p, "benign": b}[name])
    assert len(ncvardb.load_variants()) == 2


@pytest.mark.parametrize("chrom", ["M", "MT", "chrM", "chrMT"])
def test_mitochondrial_aliases_and_indels_are_counted_once(tmp_path, chrom):
    pathogenic = PATHOGENIC.replace("\tM\t", f"\t{chrom}\t").replace("\tX\t", f"\t{chrom}\t")
    table = ncvardb.read_variants(*_write(tmp_path, pathogenic=pathogenic))
    assert len(table) == 2
    assert table.attrs["n_not_snv"] == 2
    assert table.attrs["n_mitochondrial"] == 1
    assert len(table) + sum(table.attrs.values()) == 5


def test_all_excluded_returns_empty_table_with_counts(tmp_path):
    table = ncvardb.read_variants(*_write(
        tmp_path, pathogenic=PATHOGENIC.replace("\tC\tT\t", "\tN\tT\t"),
        benign=BENIGN.replace("\tG\tA\t", "\tG\tG\t")))
    assert table.empty
    assert table.attrs == {"n_not_snv": 4, "n_mitochondrial": 1}


def test_duplicate_variants_are_refused(tmp_path):
    benign = BENIGN.replace("3\t400\tG\tA", "chr1\t100\tC\tT")
    with pytest.raises(ValueError, match="duplicate"):
        ncvardb.read_variants(*_write(tmp_path, benign=benign))
