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
        # 4-track model has 256-dim embeddings
        assert OrthrusEncoder.MODEL_CONFIGS["quietflamingo/orthrus-base-4-track"]["d_model"] == 256

    def test_orthrus_6track_embedding_dim(self):
        from manylatents.dogma.encoders import OrthrusEncoder
        # 6-track model has 512-dim embeddings
        assert OrthrusEncoder.MODEL_CONFIGS["quietflamingo/orthrus-large-6-track"]["d_model"] == 512

    def test_evo2_1b_embedding_dim(self):
        from manylatents.dogma.encoders import Evo2Encoder
        assert Evo2Encoder.MODELS["evo2_1b_base"]["embedding_dim"] == 1920


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
        from manylatents.dogma.encoders.base import FoundationEncoder
        assert issubclass(ESM3Encoder, FoundationEncoder)

    def test_orthrus_inherits_foundation_encoder(self):
        from manylatents.dogma.encoders import OrthrusEncoder
        from manylatents.dogma.encoders.base import FoundationEncoder
        assert issubclass(OrthrusEncoder, FoundationEncoder)

    def test_evo2_inherits_foundation_encoder(self):
        from manylatents.dogma.encoders import Evo2Encoder
        from manylatents.dogma.encoders.base import FoundationEncoder
        assert issubclass(Evo2Encoder, FoundationEncoder)


@pytest.mark.gpu
@pytest.mark.slow
class TestESM3EncoderFunctional:
    """Functional tests for ESM3 encoder (requires GPU and model weights)."""

    @pytest.fixture
    def encoder(self):
        pytest.importorskip("esm")
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
    def encoder(self):
        pytest.importorskip("huggingface_hub")
        from manylatents.dogma.encoders import OrthrusEncoder
        return OrthrusEncoder(device="cuda")

    def test_encode_single_sequence(self, encoder):
        embedding = encoder.encode(TEST_SEQUENCES["rna"])
        assert embedding.shape[1] == 256
        assert embedding.dtype == torch.float32

    def test_encode_batch(self, encoder):
        sequences = [TEST_SEQUENCES["rna"], "AUGCAUGCAUGCAUGCAUGCAUGC"]
        embeddings = encoder.encode_batch(sequences)
        assert embeddings.shape == (2, 256)


@pytest.mark.gpu
@pytest.mark.slow
class TestEvo2EncoderFunctional:
    """Functional tests for Evo2 encoder (requires GPU and model weights)."""

    @pytest.fixture
    def encoder(self):
        pytest.importorskip("evo2")
        from manylatents.dogma.encoders import Evo2Encoder
        return Evo2Encoder(model_name="evo2_1b_base", device="cuda")

    def test_encode_single_sequence(self, encoder):
        embedding = encoder.encode(TEST_SEQUENCES["dna"])
        assert embedding.shape[1] == 1920
        assert embedding.dtype == torch.float32

    def test_encode_batch(self, encoder):
        sequences = [TEST_SEQUENCES["dna"], "ATGCATGCATGCATGCATGCATGC"]
        embeddings = encoder.encode_batch(sequences)
        assert embeddings.shape == (2, 1920)


@pytest.mark.gpu
@pytest.mark.slow
class TestCentralDogmaConsistency:
    """Test encoding the same biological information across modalities."""

    def test_encode_same_gene_all_modalities(self):
        """Encode the same gene at DNA, RNA, and protein levels."""
        pytest.importorskip("esm")
        from manylatents.dogma.encoders import ESM3Encoder, OrthrusEncoder, Evo2Encoder

        esm3 = ESM3Encoder(device="cuda")
        orthrus = OrthrusEncoder(device="cuda")
        evo2 = Evo2Encoder(model_name="evo2_1b_base", device="cuda")

        # Encode same biological information at each level
        protein_emb = esm3.encode(TEST_SEQUENCES["protein"])
        rna_emb = orthrus.encode(TEST_SEQUENCES["rna"])
        dna_emb = evo2.encode(TEST_SEQUENCES["dna"])

        # Check shapes match expected dimensions
        assert protein_emb.shape == (1, 1536), "ESM3 embedding shape mismatch"
        assert rna_emb.shape[1] == 256, "Orthrus embedding dim mismatch"
        assert dna_emb.shape[1] == 1920, "Evo2 embedding dim mismatch"

        print(f"DNA embedding shape: {dna_emb.shape}")
        print(f"RNA embedding shape: {rna_emb.shape}")
        print(f"Protein embedding shape: {protein_emb.shape}")


class TestOrthrusMultiLayerAPI:
    """Test multi-layer Orthrus API without GPU."""

    def test_default_is_single_layer(self):
        from manylatents.dogma.encoders import OrthrusEncoder
        encoder = OrthrusEncoder(device="cpu")
        assert encoder.multi_layer is False

    def test_layer_indices_enables_multi_layer(self):
        from manylatents.dogma.encoders import OrthrusEncoder
        encoder = OrthrusEncoder(device="cpu", layer_indices=[4, 7])
        assert encoder.multi_layer is True
        assert encoder.layer_indices == [4, 7]


class TestBatchEncoderMultiLayerSave:
    """Test BatchEncoder saves separate files for multi-layer output."""

    def test_multi_layer_saves_separate_files(self, tmp_path):
        from manylatents.dogma.algorithms.batch_encoder import BatchEncoder

        result = {
            "blocks.14.mlp.l3": torch.randn(5, 1920),
            "blocks.19.mlp.l3": torch.randn(5, 1920),
        }
        save_path = tmp_path / "wt_dna.pt"

        encoder = BatchEncoder.__new__(BatchEncoder)
        encoder._save_path = save_path
        encoder.datamodule = None
        encoder._save_multi_layer(result)

        assert (tmp_path / "wt_dna_blocks.14.pt").exists()
        assert (tmp_path / "wt_dna_blocks.19.pt").exists()

        loaded = torch.load(tmp_path / "wt_dna_blocks.14.pt", weights_only=False)
        assert loaded["embeddings"].shape == (5, 1920)


class TestFoundationEncoderBatchDict:
    """Test that encode_batch handles dict returns from multi-layer encoders."""

    def test_encode_batch_with_dict_encoder(self):
        """Mock encoder returning dicts should produce dict output from encode_batch."""
        from manylatents.dogma.encoders.base import FoundationEncoder

        class MockMultiLayerEncoder(FoundationEncoder):
            def __init__(self):
                super().__init__(device="cpu")
                self._embedding_dim = 4

            def _load_model(self):
                pass

            @property
            def modality(self):
                return "test"

            def encode(self, x):
                t = torch.randn(1, 4)
                return {"layer_a": t, "layer_b": t * 2}

        enc = MockMultiLayerEncoder()
        enc._model = True  # skip lazy load
        result = enc.encode_batch(["seq1", "seq2", "seq3"], batch_size=2)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"layer_a", "layer_b"}
        assert result["layer_a"].shape == (3, 4)
        assert result["layer_b"].shape == (3, 4)

    def test_encode_batch_with_tensor_encoder(self):
        """Single-layer encoder should still return flat tensor."""
        from manylatents.dogma.encoders.base import FoundationEncoder

        class MockSingleEncoder(FoundationEncoder):
            def __init__(self):
                super().__init__(device="cpu")
                self._embedding_dim = 4

            def _load_model(self):
                pass

            @property
            def modality(self):
                return "test"

            def encode(self, x):
                return torch.randn(1, 4)

        enc = MockSingleEncoder()
        enc._model = True
        result = enc.encode_batch(["seq1", "seq2"], batch_size=2)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 4)


class TestEvo2MultiLayerAPI:
    """Test multi-layer Evo2Encoder API without GPU (import/config only)."""

    def test_default_layers_is_list(self):
        from manylatents.dogma.encoders import Evo2Encoder
        encoder = Evo2Encoder(model_name="evo2_1b_base", device="cpu")
        assert isinstance(encoder.layer_names, list)
        assert len(encoder.layer_names) >= 1

    def test_single_layer_name_backward_compat(self):
        from manylatents.dogma.encoders import Evo2Encoder
        encoder = Evo2Encoder(
            model_name="evo2_1b_base",
            layer_name="blocks.14.mlp.l3",
            device="cpu",
        )
        assert encoder.layer_names == ["blocks.14.mlp.l3"]
        assert encoder.multi_layer is False

    def test_multi_layer_names(self):
        from manylatents.dogma.encoders import Evo2Encoder
        layers = ["blocks.14.mlp.l3", "blocks.19.mlp.l3", "blocks.23.mlp.l3"]
        encoder = Evo2Encoder(
            model_name="evo2_1b_base",
            layer_names=layers,
            device="cpu",
        )
        assert encoder.layer_names == layers
        assert encoder.multi_layer is True

    def test_default_1b_layers(self):
        """Default for 1B should include middle + late layers."""
        from manylatents.dogma.encoders import Evo2Encoder
        default_layers = Evo2Encoder.MODELS["evo2_1b_base"]["default_layers"]
        assert len(default_layers) == 3
        assert "blocks.14.mlp.l3" in default_layers
        assert "blocks.19.mlp.l3" in default_layers
        assert "blocks.23.mlp.l3" in default_layers


class TestEmbeddingDimConsistency:
    """Verify embedding dimensions are synchronized across the codebase.

    These tests catch stale dimension constants. If an encoder's actual output
    dim changes, ALL references must be updated together.
    """

    def test_evo2_1b_embedding_dim_is_1920(self):
        from manylatents.dogma.encoders import Evo2Encoder
        assert Evo2Encoder.MODELS["evo2_1b_base"]["embedding_dim"] == 1920

    def test_batch_encoder_dna_dim_matches_evo2(self):
        from manylatents.dogma.algorithms.batch_encoder import BatchEncoder
        from manylatents.dogma.encoders import Evo2Encoder
        expected = Evo2Encoder.MODELS["evo2_1b_base"]["embedding_dim"]
        assert BatchEncoder.DEFAULT_DIMS["dna"] == expected, (
            f"BatchEncoder dna dim ({BatchEncoder.DEFAULT_DIMS['dna']}) != "
            f"Evo2Encoder dim ({expected})"
        )

    def test_batch_encoder_protein_dim_matches_esm3(self):
        from manylatents.dogma.algorithms.batch_encoder import BatchEncoder
        assert BatchEncoder.DEFAULT_DIMS["protein"] == 1536

    def test_batch_encoder_rna_dim_matches_orthrus(self):
        from manylatents.dogma.algorithms.batch_encoder import BatchEncoder
        from manylatents.dogma.encoders import OrthrusEncoder
        expected = OrthrusEncoder.MODEL_CONFIGS["quietflamingo/orthrus-base-4-track"]["d_model"]
        assert BatchEncoder.DEFAULT_DIMS["rna"] == expected

    def test_fusion_evo2_dim_matches_encoder(self):
        from manylatents.dogma.algorithms.fusion import CentralDogmaFusion
        from manylatents.dogma.encoders import Evo2Encoder
        expected = Evo2Encoder.MODELS["evo2_1b_base"]["embedding_dim"]
        assert CentralDogmaFusion.DEFAULT_DIMS["evo2"] == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
