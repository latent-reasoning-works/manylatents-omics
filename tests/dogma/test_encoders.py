"""
Tests for central dogma foundation model encoders.

Test synthetic sequences representing the same biological information:
- DNA: ATGAAGTTTGGCGTCCGTGCCTGA
- RNA: AUGAAGUUUGGCGUCCGUGCCUGA (T→U transcription)
- Protein: MKFGVRA (translation, no stop codon for ESM3)
"""

import pytest
import torch

# Test sequences (synthetic short - 24bp)
TEST_SEQUENCES = {
    "dna": "ATGAAGTTTGGCGTCCGTGCCTGA",
    "rna": "AUGAAGUUUGGCGUCCGUGCCUGA",
    "protein": "MKFGVRA",
}

# GFP sequences for longer validation
GFP_SEQUENCES = {
    "dna": "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA",
    "protein": "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
}


class TestEncoderImports:
    """Test that encoders can be imported without loading models."""

    def test_esm3_encoder_import(self):
        from manylatents.dogma.encoders import ESM3Encoder
        assert ESM3Encoder is not None

    def test_orthrus_encoder_import(self):
        from manylatents.dogma.encoders import OrthrusEncoder
        assert OrthrusEncoder is not None

    def test_evo2_encoder_import(self):
        from manylatents.dogma.encoders import Evo2Encoder
        assert Evo2Encoder is not None


class TestEncoderProperties:
    """Test encoder properties without loading models."""

    def test_esm3_embedding_dim(self):
        from manylatents.dogma.encoders import ESM3Encoder
        encoder = ESM3Encoder.__new__(ESM3Encoder)
        encoder._embedding_dim = 1536
        assert encoder.embedding_dim == 1536

    def test_orthrus_4track_embedding_dim(self):
        from manylatents.dogma.encoders import OrthrusEncoder
        assert OrthrusEncoder.MODELS[4]["embedding_dim"] == 256

    def test_orthrus_6track_embedding_dim(self):
        from manylatents.dogma.encoders import OrthrusEncoder
        assert OrthrusEncoder.MODELS[6]["embedding_dim"] == 512

    def test_evo2_1b_embedding_dim(self):
        from manylatents.dogma.encoders import Evo2Encoder
        assert Evo2Encoder.MODELS["evo2_1b_base"]["embedding_dim"] == 2048


class TestEncoderModality:
    """Test that encoders report correct modalities."""

    def test_esm3_modality(self):
        from manylatents.dogma.encoders import ESM3Encoder
        encoder = ESM3Encoder.__new__(ESM3Encoder)
        assert encoder.modality == "protein"

    def test_orthrus_modality(self):
        from manylatents.dogma.encoders import OrthrusEncoder
        encoder = OrthrusEncoder.__new__(OrthrusEncoder)
        assert encoder.modality == "rna"

    def test_evo2_modality(self):
        from manylatents.dogma.encoders import Evo2Encoder
        encoder = Evo2Encoder.__new__(Evo2Encoder)
        assert encoder.modality == "dna"


class TestFoundationEncoderInterface:
    """Test that encoders inherit from FoundationEncoder correctly."""

    def test_esm3_inherits_foundation_encoder(self):
        from manylatents.dogma.encoders import ESM3Encoder
        from manylatents.algorithms.encoder import FoundationEncoder
        assert issubclass(ESM3Encoder, FoundationEncoder)

    def test_orthrus_inherits_foundation_encoder(self):
        from manylatents.dogma.encoders import OrthrusEncoder
        from manylatents.algorithms.encoder import FoundationEncoder
        assert issubclass(OrthrusEncoder, FoundationEncoder)

    def test_evo2_inherits_foundation_encoder(self):
        from manylatents.dogma.encoders import Evo2Encoder
        from manylatents.algorithms.encoder import FoundationEncoder
        assert issubclass(Evo2Encoder, FoundationEncoder)


@pytest.mark.gpu
@pytest.mark.slow
class TestESM3EncoderFunctional:
    """Functional tests for ESM3 encoder (requires GPU and model weights)."""

    @pytest.fixture
    def encoder(self):
        from manylatents.dogma.encoders import ESM3Encoder
        return ESM3Encoder(device="cuda")

    def test_encode_single_sequence(self, encoder):
        embedding = encoder.encode(TEST_SEQUENCES["protein"])
        assert embedding.shape == (1, 1536)
        assert embedding.dtype == torch.float32
        assert embedding.device.type == "cuda"

    def test_encode_batch(self, encoder):
        sequences = [TEST_SEQUENCES["protein"], "MKTAYIAKQRQISFVKSHFSRQLE"]
        embeddings = encoder.encode_batch(sequences)
        assert embeddings.shape == (2, 1536)

    def test_encode_gfp(self, encoder):
        embedding = encoder.encode(GFP_SEQUENCES["protein"])
        assert embedding.shape == (1, 1536)


@pytest.mark.gpu
@pytest.mark.slow
class TestOrthrusEncoderFunctional:
    """Functional tests for Orthrus encoder (requires GPU and model weights)."""

    @pytest.fixture
    def encoder_4track(self):
        from manylatents.dogma.encoders import OrthrusEncoder
        return OrthrusEncoder(n_tracks=4, device="cuda")

    def test_encode_single_sequence(self, encoder_4track):
        embedding = encoder_4track.encode(TEST_SEQUENCES["rna"])
        assert embedding.shape[1] == 256
        assert embedding.dtype == torch.float32

    def test_encode_batch(self, encoder_4track):
        sequences = [TEST_SEQUENCES["rna"], "AUGCAUGCAUGCAUGCAUGCAUGC"]
        embeddings = encoder_4track.encode_batch(sequences)
        assert embeddings.shape == (2, 256)


@pytest.mark.gpu
@pytest.mark.slow
class TestEvo2EncoderFunctional:
    """Functional tests for Evo2 encoder (requires GPU and model weights)."""

    @pytest.fixture
    def encoder(self):
        from manylatents.dogma.encoders import Evo2Encoder
        return Evo2Encoder(model_name="evo2_1b_base", device="cuda")

    def test_encode_single_sequence(self, encoder):
        embedding = encoder.encode(TEST_SEQUENCES["dna"])
        assert embedding.shape[1] == 2048
        assert embedding.dtype == torch.float32

    def test_encode_batch(self, encoder):
        sequences = [TEST_SEQUENCES["dna"], "ATGCATGCATGCATGCATGCATGC"]
        embeddings = encoder.encode_batch(sequences)
        assert embeddings.shape == (2, 2048)


@pytest.mark.gpu
@pytest.mark.slow
class TestCentralDogmaConsistency:
    """Test encoding the same biological information across modalities."""

    def test_encode_same_gene_all_modalities(self):
        """Encode the same gene at DNA, RNA, and protein levels."""
        from manylatents.dogma.encoders import ESM3Encoder, OrthrusEncoder, Evo2Encoder

        esm3 = ESM3Encoder(device="cuda")
        orthrus = OrthrusEncoder(n_tracks=4, device="cuda")
        evo2 = Evo2Encoder(model_name="evo2_1b_base", device="cuda")

        # Encode same biological information at each level
        protein_emb = esm3.encode(TEST_SEQUENCES["protein"])
        rna_emb = orthrus.encode(TEST_SEQUENCES["rna"])
        dna_emb = evo2.encode(TEST_SEQUENCES["dna"])

        # Check shapes match expected dimensions
        assert protein_emb.shape == (1, 1536), "ESM3 embedding shape mismatch"
        assert rna_emb.shape[1] == 256, "Orthrus embedding dim mismatch"
        assert dna_emb.shape[1] == 2048, "Evo2 embedding dim mismatch"

        print(f"DNA embedding shape: {dna_emb.shape}")
        print(f"RNA embedding shape: {rna_emb.shape}")
        print(f"Protein embedding shape: {protein_emb.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
