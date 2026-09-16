"""The ClinVar staging adapter's schema mapping, on a fixture rather than 22 GB."""

import csv
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "stage_clinvar.py"

BUILT_HEADER = ["variation_id", "chrom", "pos", "ref", "alt", "gene",
                "clinical_significance", "label"]
BUILT_ROWS = [
    ["2193183", "1", "12345", "A", "G", "BRCA1", "Pathogenic", "1"],
    ["2193184", "X", "999", "C", "T", "brca2", "Benign", "0"],
]


def _build(tmp_path: Path, consequence: str = "missense") -> Path:
    source = tmp_path / "clinvar"
    (source / "variants").mkdir(parents=True)
    (source / "sequences").mkdir(parents=True)
    with (source / "variants" / f"{consequence}.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(BUILT_HEADER)
        writer.writerows(BUILT_ROWS)
    (source / "sequences" / f"{consequence}_dna.fasta").write_text(">clinvar_2193183\nACGT\n")
    (source / "sequences" / f"{consequence}_dna.fasta.fai").write_text("clinvar_2193183\t4\t17\t4\t5\n")
    return source


def _run(source: Path, dest: Path, consequence: str = "missense"):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(source), "--dest", str(dest),
         "--consequence", consequence],
        capture_output=True, text=True,
    )


def test_columns_are_renamed_to_what_the_loader_reads(tmp_path: Path):
    """The loader indexes gene_symbol/chromosome/start/stop; the build has
    gene/chrom/pos. A symlink would leave every lookup with a KeyError."""
    dest = tmp_path / "staged"
    assert _run(_build(tmp_path), dest).returncode == 0
    with (dest / "variants.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["gene_symbol"] == "BRCA1"
    assert rows[0]["chromosome"] == "1"
    assert rows[0]["label"] == "1"
    assert rows[1]["gene_symbol"] == "brca2"


def test_point_substitutions_get_equal_start_and_stop(tmp_path: Path):
    dest = tmp_path / "staged"
    _run(_build(tmp_path), dest)
    with (dest / "variants.tsv").open() as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["start"] == row["stop"] == "12345"


def test_absent_fields_get_explicit_sentinels(tmp_path: Path):
    """review_status and variant_type are not in the build. They must be marked
    absent, never invented, since they land in the loader's metadata."""
    dest = tmp_path / "staged"
    _run(_build(tmp_path), dest)
    with (dest / "variants.tsv").open() as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["review_status"] == "not_provided"
    assert row["variant_type"] == "single_nucleotide_variant"


def test_fasta_and_index_are_linked_together(tmp_path: Path):
    """The .fai must stay beside its fasta or random access breaks."""
    dest = tmp_path / "staged"
    _run(_build(tmp_path), dest)
    assert (dest / "dna.fasta").is_symlink()
    assert (dest / "dna.fasta.fai").is_symlink()
    assert (dest / "dna.fasta").read_text().startswith(">clinvar_")


def test_missing_consequence_class_is_refused(tmp_path: Path):
    result = _run(_build(tmp_path), tmp_path / "staged", consequence="nonsense")
    assert result.returncode != 0
    assert "no such consequence table" in result.stdout + result.stderr
