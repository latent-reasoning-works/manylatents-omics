"""Tests for AlphaGenomeEncoder."""

import pytest
import torch


class TestAlphaGenomeEncoderUnit:
    """Unit tests for AlphaGenomeEncoder (no model loading)."""

    def test_modality_is_dna(self):
        """modality property should return 'dna'."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cpu")
        assert encoder.modality == "dna"

    def test_embedding_dim(self):
        """embedding_dim should match MODELS config."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(model_name="alphagenome", device="cpu")
        assert encoder.embedding_dim == 1536

        encoder_128bp = AlphaGenomeEncoder(model_name="alphagenome_128bp", device="cpu")
        assert encoder_128bp.embedding_dim == 3072

    def test_context_length(self):
        """context_length should be 1M bp."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cpu")
        assert encoder.context_length == 1_000_000

    def test_invalid_model_name(self):
        """Should raise ValueError for invalid model_name."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        with pytest.raises(ValueError, match="model_name must be one of"):
            AlphaGenomeEncoder(model_name="invalid_model", device="cpu")

    def test_split_sequence(self):
        """_split_sequence should create overlapping chunks."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        seq = "ATGC" * 100  # 400 bp
        chunks = AlphaGenomeEncoder._split_sequence(seq, chunk_size=100, overlap=20)

        assert len(chunks) == 5  # 400 bp / (100-20) = 5 chunks
        assert all(len(c) <= 100 for c in chunks)

    def test_split_sequence_overlap_error(self):
        """Should raise ValueError if overlap >= chunk_size."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        with pytest.raises(ValueError, match="overlap .* must be < chunk_size"):
            AlphaGenomeEncoder._split_sequence("ATGC", chunk_size=100, overlap=100)

    def test_supported_lengths_match_public_client(self):
        """SUPPORTED_LENGTHS should mirror what the AlphaGenome client permits."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        assert AlphaGenomeEncoder.SUPPORTED_LENGTHS == (
            16_384,
            131_072,
            524_288,
            1_048_576,
        )

    def test_encode_layers_requires_layers(self):
        """encode_layers() should reject an empty layer list before loading."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cpu")
        with pytest.raises(ValueError, match="at least one module"):
            encoder.encode_layers("ACGT" * 4096, layers=[])

    def test_encode_layers_warns_on_untrained_length(self):
        """A sequence shorter than 16 kb is an extrapolation and must say so.

        This is the guard for the matched-envelope control: shrinking the input
        is not the way to match a smaller receptive field.
        """
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cpu")
        # 2114 bp is Cherimoya's window — the length the crop control proposed.
        with pytest.warns(RuntimeWarning, match="not one of AlphaGenome's"):
            with pytest.raises(Exception):
                # Loading real weights will fail in unit-test context; the
                # warning must already have fired by then.
                encoder.encode_layers("A" * 2114, layers=["alphagenome"])


def test_import_from_encoders():
    """AlphaGenomeEncoder should be importable from encoders module."""
    from manylatents.dogma.encoders import AlphaGenomeEncoder

    assert AlphaGenomeEncoder is not None


@pytest.mark.gpu
@pytest.mark.slow
class TestAlphaGenomeEncoderGPU:
    """Integration tests requiring GPU and model weights.

    These tests download model weights from HuggingFace on first run.
    Run with: pytest -m gpu tests/dogma/encoders/test_alphagenome.py
    """

    def test_encode_returns_tensor(self):
        """encode() should return a PyTorch tensor with correct shape."""
        pytest.importorskip("jax")
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cuda")
        # Short sequence for speed
        sequence = "ATGAAGTTTGGCGTCCGTGCCTGA" * 10  # 240 bp

        embedding = encoder.encode(sequence)

        assert isinstance(embedding, torch.Tensor)
        assert embedding.device.type == "cuda"
        assert embedding.shape == (1, encoder.embedding_dim)
        assert not torch.isnan(embedding).any()

    def test_encode_with_aggregation(self):
        """encode() with chunking should aggregate correctly."""
        pytest.importorskip("jax")
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cuda")
        long_seq = "ATGC" * 1000  # 4000 bp

        # Mean aggregation (default)
        result_mean = encoder.encode(long_seq, chunk_size=1000, overlap=100, aggregate="mean")
        assert result_mean.shape == (1, encoder.embedding_dim)

        # Max aggregation
        result_max = encoder.encode(long_seq, chunk_size=1000, overlap=100, aggregate="max")
        assert result_max.shape == (1, encoder.embedding_dim)

        # No aggregation
        result_chunks = encoder.encode(long_seq, chunk_size=1000, overlap=100, aggregate=None)
        assert result_chunks.shape[0] > 1  # Multiple chunks
        assert result_chunks.shape[1] == encoder.embedding_dim

    def test_predict_returns_dict(self):
        """predict() should return dict of regulatory track tensors."""
        pytest.importorskip("jax")
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cuda")
        sequence = "ATGAAGTTTGGCGTCCGTGCCTGA" * 10

        predictions = encoder.predict(sequence)

        assert isinstance(predictions, dict)
        assert len(predictions) > 0

        for name, tensor in predictions.items():
            assert isinstance(name, str)
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.type == "cuda"
            assert not torch.isnan(tensor).any()

    def test_predict_specific_outputs(self):
        """predict() should return only requested output types."""
        pytest.importorskip("jax")
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cuda")
        sequence = "ATGAAGTTTGGCGTCCGTGCCTGA" * 10

        # Request only ATAC and DNASE
        predictions = encoder.predict(sequence, output_types=["ATAC", "DNASE"])

        assert "atac" in predictions or "dnase" in predictions
        # Should not have unrequested outputs
        assert "cage" not in predictions or "rna_seq" not in predictions

    def test_available_layers_exposes_transformer_tower(self):
        """available_layers() should enumerate the nine transformer blocks."""
        pytest.importorskip("jax")
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cuda")
        layers = encoder.available_layers(sequence_length=16_384)

        blocks = [n for n in layers if "transformer_tower/mha_block" in n]
        assert len(blocks) == 9, f"expected 9 transformer blocks, got {blocks}"
        # Trunk is pooled 128x from the input.
        assert layers[blocks[0]] == [1, 128, 1536]

    def test_encode_layers_returns_requested_modules(self):
        """encode_layers() should tap named modules and pool to (1, D)."""
        pytest.importorskip("jax")
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cuda")
        wanted = [
            "alphagenome/transformer_tower/mha_block",
            "alphagenome/transformer_tower/mha_block_8",
        ]
        acts = encoder.encode_layers("ACGT" * 4096, layers=wanted)  # 16384 bp

        assert set(acts) == set(wanted)
        for name, tensor in acts.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (1, 1536), f"{name}: {tensor.shape}"
            assert not torch.isnan(tensor).any()

    def test_encode_layers_rejects_unknown_module(self):
        """An unreachable module name should fail loudly, not silently."""
        pytest.importorskip("jax")
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cuda")
        with pytest.raises(KeyError, match="not reached"):
            encoder.encode_layers("ACGT" * 4096, layers=["no/such/module"])
