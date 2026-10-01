"""Track-scoring preparation without loading or running a model."""
import subprocess
import sys

import pandas as pd
import pytest

from manylatents.dogma.score_tracks import prefilter_variants, shard_slice


@pytest.mark.parametrize("n,n_shards", [(0, 3), (1, 4), (10, 3), (12, 4), (17, 1)])
def test_shards_cover_rows(n, n_shards):
    rows = list(range(n))
    assert [row for shard in range(n_shards)
            for row in rows[shard_slice(n, shard, n_shards)]] == rows


@pytest.mark.parametrize("args", [(-1, 0, 1), (2, 0, 0), (2, -1, 2), (2, 2, 2)])
def test_invalid_shard(args):
    with pytest.raises(ValueError):
        shard_slice(*args)


def test_prefilter(tmp_path):
    pytest.importorskip("pyfaidx")
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nACGTACGTACGTACGTACGT\n")
    variants = pd.DataFrame({"chromosome": ["chr1", "chr1", "chr1", "chr2"],
                             "pos": [9, 10, 20, 9], "ref": ["a", "A", "T", "A"],
                             "alt": ["G"] * 4})
    original = variants.copy()
    kept, dropped = prefilter_variants(variants, str(fasta), 8)
    assert kept.index.tolist() == [0]
    assert dropped.reason.tolist() == ["reference_mismatch", "window_off_chromosome",
                                        "missing_chromosome"]
    pd.testing.assert_frame_equal(original, variants)
    kept, dropped = prefilter_variants(variants.iloc[:0], str(fasta), 8)
    assert kept.empty and dropped.empty and "reason" in dropped


def test_help_without_heavy_imports():
    code = '''
import sys
class BlockHeavy:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"jax", "alphagenome", "alphagenome_research", "huggingface_hub"}:
            raise AssertionError("unexpected heavy import: " + fullname)
sys.meta_path.insert(0, BlockHeavy())
from manylatents.dogma.score_tracks import main
try:
    main(["--help"])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("help did not exit")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--fasta FASTA" in result.stdout


def test_empty_shard_does_not_load_model(tmp_path):
    pytest.importorskip("pyfaidx")
    from manylatents.dogma.score_tracks import main

    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nACGT\n")
    variants = tmp_path / "variants.tsv"
    pd.DataFrame({"chrom": ["1"], "pos": [1], "ref": ["A"], "alt": ["G"]}).to_csv(
        variants, sep="\t", index=False)
    assert main(["--variants", str(variants), "--out", str(tmp_path / "out"),
                 "--fasta", str(fasta)]) == 0


def test_prefilter_window_boundaries(tmp_path):
    pytest.importorskip("pyfaidx")
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "A" * 20 + "\n")
    # Even windows around SNVs use the rounded-up reference interval centre.
    variants = pd.DataFrame({"chromosome": ["chr1"] * 4, "pos": [4, 16, 17, 18],
                             "ref": ["A"] * 4})
    kept, dropped = prefilter_variants(variants, str(fasta), 8)
    assert kept.pos.tolist() == [4, 16]
    assert dropped.reason.tolist() == ["window_off_chromosome"] * 2
    kept, _ = prefilter_variants(variants, str(fasta), 7)
    assert kept.pos.tolist() == [4, 16, 17]
