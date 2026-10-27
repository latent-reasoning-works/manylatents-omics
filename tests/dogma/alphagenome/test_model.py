"""Input construction for AlphaGenome: window placement, alleles and one-hot."""
import numpy as np
import pytest

from manylatents.dogma.alphagenome.model import (
    SUPPORTED_LENGTHS, LocalModel, allele_bases, one_hot, other_bases, site_sequences, site_window,
)


@pytest.mark.parametrize("length", sorted(SUPPORTED_LENGTHS))
def test_snv_lands_at_half_length_minus_one(length):
    start, index = site_window(1_000_000, 1, length)
    assert index == length // 2 - 1
    assert start + index == 1_000_000 - 1


def test_one_megabase_index():
    assert site_window(5_000_000, 1, 1_048_576)[1] == 524_287


@pytest.mark.parametrize("pos,ref_length,length", [(100, 1, 8), (100, 2, 8), (100, 3, 9), (7, 1, 7)])
def test_window_matches_alphagenome_resize(pos, ref_length, length):
    # genome.Interval.resize keeps the centre, rounded up for an unstranded interval
    start0, end0 = pos - 1, pos - 1 + ref_length
    centre = (start0 + end0 + 1) // 2
    expected_start = centre - (length + 1) // 2
    assert site_window(pos, ref_length, length) == (expected_start, start0 - expected_start)


def test_one_hot_matches_alphagenome_table():
    out = one_hot("ACGTNacgtn")
    assert out.dtype == np.float32 and out.shape == (10, 4)
    np.testing.assert_array_equal(out[:4], np.eye(4))
    np.testing.assert_array_equal(out[5:9], np.eye(4))
    assert not out[4].any() and not out[9].any()


def test_allele_bases():
    assert other_bases("C", "T") == ("A", "G")
    assert allele_bases("g", "a") == {"ref": "G", "alt": "A", "alt2": "C", "alt3": "T"}
    assert allele_bases("G", "A", "ref_alt") == {"ref": "G", "alt": "A"}
    with pytest.raises(ValueError):
        allele_bases("A", "A")
    with pytest.raises(ValueError):
        allele_bases("A", "C", "three")


def test_site_sequences():
    genome = {"chr1": "AACCGGTTAC" * 3}
    sites = site_sequences("chr1", 11, "A", "G", genome, 8)
    start, index = site_window(11, 1, 8)
    assert (sites.start, sites.site_index) == (start, index) == (7, 3)
    assert sites.reference_matches and sites.reference_base == "A"
    window = genome["chr1"][7:15]
    assert sites.sequences["ref"] == window
    for allele, base in sites.bases.items():
        seq = sites.sequences[allele]
        assert len(seq) == 8 and seq[index] == base
        assert seq[:index] + seq[index + 1:] == window[:index] + window[index + 1:]
    assert sorted(sites.bases.values()) == list("ACGT")


def test_site_sequences_mismatch_and_errors():
    genome = {"chr1": "ACGTACGTACGTACGT"}
    sites = site_sequences("chr1", 9, "C", "T", genome, 4)
    assert not sites.reference_matches and sites.reference_base == "A"
    assert sites.sequences["ref"][sites.site_index] == "A"
    with pytest.raises(ValueError, match="leaves"):
        site_sequences("chr1", 1, "A", "C", genome, 4)
    with pytest.raises(ValueError, match="not an SNV"):
        site_sequences("chr1", 9, "AC", "T", genome, 4)


def test_check_length():
    model = LocalModel(params={}, state={}, forward=None, supported_lengths=frozenset({512}))
    model.check_length(512)
    with pytest.raises(ValueError, match="trained lengths"):
        model.check_length(1000)
    assert model.organism_index == 0
    assert LocalModel(params={}, state={}, forward=None, organism="MUS_MUSCULUS").organism_index == 1


@pytest.mark.gpu
def test_load_local_real_weights():
    pytest.importorskip("alphagenome_research")
    from manylatents.dogma.alphagenome.model import load_local
    from manylatents.dogma.alphagenome.taps import DEFAULT_TRUNK_LAYERS, available_layers

    try:
        model = load_local()
    except Exception as err:  # noqa: BLE001 - no cached weights here
        pytest.skip(f"AlphaGenome weights not available: {err}")
    seen = available_layers(model, 16_384)
    for spec in DEFAULT_TRUNK_LAYERS:
        assert spec.module in seen
