"""Tests for VariantDataModule - paired WT/MUT sequence loading."""
import pytest
from pathlib import Path

# Data directory for existing ClinVar files
DATA_DIR = Path(__file__).parents[5] / "data" / "clinvar"


class TestVariantDataModuleLoading:
    """Test VariantDataModule loads paired sequences correctly."""

    def test_variant_datamodule_loads_paired_sequences(self):
        """VariantDataModule should load WT and MUT sequences for a variant type."""
        from manylatents.dogma.data.variant_datamodule import VariantDataModule

        dm = VariantDataModule(
            variants_dir=DATA_DIR / "variants",
            sequences_dir=DATA_DIR / "sequences",
            variant_type="synonymous",
            modality="dna",
            max_variants=10,
        )
        dm.setup()

        pairs = dm.get_sequence_pairs()
        assert len(pairs["wt"]) == len(pairs["mut"]) == 10
        assert all(isinstance(s, str) for s in pairs["wt"])
        assert all(isinstance(s, str) for s in pairs["mut"])

    def test_variant_datamodule_injects_variant(self):
        """MUT sequence should differ from WT at variant position."""
        from manylatents.dogma.data.variant_datamodule import VariantDataModule

        dm = VariantDataModule(
            variants_dir=DATA_DIR / "variants",
            sequences_dir=DATA_DIR / "sequences",
            variant_type="synonymous",
            modality="dna",
            max_variants=5,
        )
        dm.setup()

        pairs = dm.get_sequence_pairs()
        # At least some should differ (synonymous DNA still has nucleotide change)
        n_different = sum(w != m for w, m in zip(pairs["wt"], pairs["mut"]))
        assert n_different > 0, "MUT sequences should differ from WT"

    def test_get_labels_returns_correct_shape(self):
        """Labels should match number of loaded variants."""
        from manylatents.dogma.data.variant_datamodule import VariantDataModule

        dm = VariantDataModule(
            variants_dir=DATA_DIR / "variants",
            sequences_dir=DATA_DIR / "sequences",
            variant_type="synonymous",
            modality="dna",
            max_variants=10,
        )
        dm.setup()

        labels = dm.get_labels()
        assert len(labels) == 10

    def test_get_variant_ids_returns_list(self):
        """Variant IDs should be returned as a list of strings."""
        from manylatents.dogma.data.variant_datamodule import VariantDataModule

        dm = VariantDataModule(
            variants_dir=DATA_DIR / "variants",
            sequences_dir=DATA_DIR / "sequences",
            variant_type="synonymous",
            modality="dna",
            max_variants=5,
        )
        dm.setup()

        variant_ids = dm.get_variant_ids()
        assert len(variant_ids) == 5
        assert all(isinstance(vid, str) for vid in variant_ids)
        assert all(vid.startswith("clinvar_") for vid in variant_ids)


class TestVariantDataModuleRNAModality:
    """Test RNA modality handling."""

    def test_rna_modality_loads(self):
        """RNA modality should load sequences from RNA FASTA.

        Note: RNA sequences are stored in DNA alphabet (T not U) for compatibility
        with Orthrus encoder which expects DNA alphabet input.
        """
        from manylatents.dogma.data.variant_datamodule import VariantDataModule

        dm = VariantDataModule(
            variants_dir=DATA_DIR / "variants",
            sequences_dir=DATA_DIR / "sequences",
            variant_type="synonymous",
            modality="rna",
            max_variants=5,
        )
        dm.setup()

        pairs = dm.get_sequence_pairs()
        # Verify we loaded sequences
        assert len(pairs["wt"]) == 5
        assert all(seq for seq in pairs["wt"])  # Non-empty


class TestVariantDataModuleProteinModality:
    """Test protein modality handling."""

    @pytest.mark.skipif(
        not (DATA_DIR / "sequences" / "synonymous_protein.fasta").exists(),
        reason="Protein FASTA not available",
    )
    def test_protein_modality_loads(self):
        """Protein modality should load amino acid sequences."""
        from manylatents.dogma.data.variant_datamodule import VariantDataModule

        dm = VariantDataModule(
            variants_dir=DATA_DIR / "variants",
            sequences_dir=DATA_DIR / "sequences",
            variant_type="synonymous",
            modality="protein",
            max_variants=5,
        )
        dm.setup()

        pairs = dm.get_sequence_pairs()
        # For synonymous variants, WT and MUT protein should be SAME
        # (that's what synonymous means - same amino acid)
        for wt, mut in zip(pairs["wt"], pairs["mut"]):
            if wt and mut:
                assert wt == mut, "Synonymous variants should have same protein"


class TestVariantDataModuleEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_variant_type_raises(self):
        """Should raise if variant type TSV doesn't exist."""
        from manylatents.dogma.data.variant_datamodule import VariantDataModule

        dm = VariantDataModule(
            variants_dir=DATA_DIR / "variants",
            sequences_dir=DATA_DIR / "sequences",
            variant_type="nonexistent_type",
            modality="dna",
        )
        with pytest.raises(FileNotFoundError):
            dm.setup()

    def test_empty_max_variants_loads_all(self):
        """max_variants=None should load all variants."""
        from manylatents.dogma.data.variant_datamodule import VariantDataModule

        dm = VariantDataModule(
            variants_dir=DATA_DIR / "variants",
            sequences_dir=DATA_DIR / "sequences",
            variant_type="synonymous",
            modality="dna",
            max_variants=None,
        )
        dm.setup()

        # Should load more than 10 (we know synonymous has many)
        assert len(dm.get_sequence_pairs()["wt"]) > 10


class TestVariantDataModuleBatchEncoderCompatibility:
    """Test compatibility with BatchEncoder via get_sequences()."""

    def test_get_sequences_returns_wt_mut_channels(self):
        """get_sequences() should return dict with 'wt' and 'mut' keys."""
        from manylatents.dogma.data.variant_datamodule import VariantDataModule

        dm = VariantDataModule(
            variants_dir=DATA_DIR / "variants",
            sequences_dir=DATA_DIR / "sequences",
            variant_type="synonymous",
            modality="dna",
            max_variants=5,
        )
        dm.setup()

        seqs = dm.get_sequences()
        assert "wt" in seqs
        assert "mut" in seqs
        assert len(seqs["wt"]) == 5
        assert len(seqs["mut"]) == 5

    def test_get_sequences_matches_get_sequence_pairs(self):
        """get_sequences() should return same data as get_sequence_pairs()."""
        from manylatents.dogma.data.variant_datamodule import VariantDataModule

        dm = VariantDataModule(
            variants_dir=DATA_DIR / "variants",
            sequences_dir=DATA_DIR / "sequences",
            variant_type="synonymous",
            modality="dna",
            max_variants=5,
        )
        dm.setup()

        seqs = dm.get_sequences()
        pairs = dm.get_sequence_pairs()

        assert seqs["wt"] == pairs["wt"]
        assert seqs["mut"] == pairs["mut"]
