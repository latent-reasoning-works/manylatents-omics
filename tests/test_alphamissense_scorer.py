"""Tests for AlphaMissenseScorer."""

import gzip
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manylatents.dogma.scorers.alphamissense import AlphaMissenseScorer


@pytest.fixture
def genomic_tsv(tmp_path):
    """Create a minimal AlphaMissense hg38 TSV for testing."""
    content = """\
# AlphaMissense (test fixture)
# Cheng et al., Science 2023
#CHROM	POS	REF	ALT	genome	uniprot_id	transcript_id	protein_variant	am_pathogenicity	am_class
1	69094	G	A	hg38	A0A024RBG1	ENST00000335137.4	G23E	0.0833	likely_benign
1	69094	G	C	hg38	A0A024RBG1	ENST00000335137.4	G23A	0.0673	likely_benign
17	43045726	T	C	hg38	P38398	ENST00000357654.9	V1736A	0.9421	likely_pathogenic
17	43045726	T	G	hg38	P38398	ENST00000357654.9	V1736G	0.8812	likely_pathogenic
"""
    path = tmp_path / "AlphaMissense_hg38.tsv.gz"
    with gzip.open(path, "wt") as f:
        f.write(content)
    return path


@pytest.fixture
def protein_tsv(tmp_path):
    """Create a minimal AlphaMissense protein substitutions TSV."""
    content = """\
# AlphaMissense (test fixture)
# Cheng et al., Science 2023
uniprot_id	protein_variant	am_pathogenicity	am_class
P38398	V1736A	0.9421	likely_pathogenic
P38398	V1736G	0.8812	likely_pathogenic
A0A024RBG1	G23E	0.0833	likely_benign
"""
    path = tmp_path / "AlphaMissense_aa_substitutions.tsv.gz"
    with gzip.open(path, "wt") as f:
        f.write(content)
    return path


class TestAlphaMissenseScorer:
    def test_genomic_scoring(self, genomic_tsv):
        scorer = AlphaMissenseScorer(genomic_tsv, match_by="genomic")
        variants = pd.DataFrame({
            "CHROM": ["1", "17", "X"],
            "POS": [69094, 43045726, 100000],
            "REF": ["G", "T", "A"],
            "ALT": ["A", "C", "G"],
        })
        scores = scorer.score(variants)
        assert len(scores) == 3
        np.testing.assert_almost_equal(scores[0], 0.0833, decimal=3)
        np.testing.assert_almost_equal(scores[1], 0.9421, decimal=3)
        assert np.isnan(scores[2])  # not in AlphaMissense

    def test_genomic_with_chr_prefix(self, genomic_tsv):
        """Handles 'chr1' vs '1' mismatch."""
        scorer = AlphaMissenseScorer(genomic_tsv, match_by="genomic")
        variants = pd.DataFrame({
            "CHROM": ["chr1", "chr17"],
            "POS": [69094, 43045726],
            "REF": ["G", "T"],
            "ALT": ["A", "C"],
        })
        scores = scorer.score(variants)
        np.testing.assert_almost_equal(scores[0], 0.0833, decimal=3)
        np.testing.assert_almost_equal(scores[1], 0.9421, decimal=3)

    def test_clinvar_column_names(self, genomic_tsv):
        """Works with ClinVarDataModule metadata column names."""
        scorer = AlphaMissenseScorer(genomic_tsv, match_by="genomic")
        variants = pd.DataFrame({
            "chromosome": ["1", "17"],
            "start": [69094, 43045726],
            "ref": ["G", "T"],
            "alt": ["A", "C"],
        })
        scores = scorer.score(variants)
        np.testing.assert_almost_equal(scores[0], 0.0833, decimal=3)
        np.testing.assert_almost_equal(scores[1], 0.9421, decimal=3)

    def test_protein_scoring(self, protein_tsv):
        scorer = AlphaMissenseScorer(protein_tsv, match_by="protein")
        variants = pd.DataFrame({
            "uniprot_id": ["P38398", "P38398", "Q99999"],
            "protein_variant": ["V1736A", "V1736G", "A1B"],
        })
        scores = scorer.score(variants)
        np.testing.assert_almost_equal(scores[0], 0.9421, decimal=3)
        np.testing.assert_almost_equal(scores[1], 0.8812, decimal=3)
        assert np.isnan(scores[2])

    def test_score_from_keys(self, genomic_tsv):
        scorer = AlphaMissenseScorer(genomic_tsv, match_by="genomic")
        scores = scorer.score_from_keys(
            chroms=["1", "17"],
            positions=[69094, 43045726],
            refs=["G", "T"],
            alts=["A", "C"],
        )
        np.testing.assert_almost_equal(scores[0], 0.0833, decimal=3)
        np.testing.assert_almost_equal(scores[1], 0.9421, decimal=3)

    def test_score_from_keys_protein_mode_raises(self, protein_tsv):
        scorer = AlphaMissenseScorer(protein_tsv, match_by="protein")
        with pytest.raises(ValueError, match="score_from_keys only works"):
            scorer.score_from_keys(["1"], [100], ["A"], ["G"])

    def test_invalid_match_by(self, genomic_tsv):
        with pytest.raises(ValueError, match="match_by must be"):
            AlphaMissenseScorer(genomic_tsv, match_by="invalid")

    def test_lazy_loading(self, genomic_tsv):
        scorer = AlphaMissenseScorer(genomic_tsv)
        assert scorer._lookup is None
        scorer.score(pd.DataFrame({"CHROM": [], "POS": [], "REF": [], "ALT": []}))
        assert scorer._lookup is not None

    def test_repr(self, genomic_tsv):
        scorer = AlphaMissenseScorer(genomic_tsv)
        assert "not loaded" in repr(scorer)
        scorer._load()
        assert "4" in repr(scorer)  # 4 entries in fixture

    def test_all_missing_returns_nan(self, genomic_tsv):
        scorer = AlphaMissenseScorer(genomic_tsv)
        variants = pd.DataFrame({
            "CHROM": ["X", "Y"],
            "POS": [1, 2],
            "REF": ["A", "C"],
            "ALT": ["G", "T"],
        })
        scores = scorer.score(variants)
        assert np.all(np.isnan(scores))

    def test_empty_input(self, genomic_tsv):
        scorer = AlphaMissenseScorer(genomic_tsv)
        scores = scorer.score(pd.DataFrame({"CHROM": [], "POS": [], "REF": [], "ALT": []}))
        assert len(scores) == 0
