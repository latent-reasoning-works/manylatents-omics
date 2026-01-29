"""Tests for AlphaGenomeEncoder."""

import pytest
import torch


class TestAlphaGenomeEncoder:
    """Unit tests for AlphaGenomeEncoder."""

    def test_encode_returns_tensor(self):
        """encode() should return a PyTorch tensor."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cpu")
        # Short sequence, no chunking needed
        result = encoder.encode("ATGC" * 100)

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1  # batch dim
        assert result.shape[1] == encoder.embedding_dim

    def test_encode_with_chunking(self):
        """encode() should handle chunked sequences."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cpu")
        long_seq = "ATGC" * 1000  # 4000 bp

        # With aggregation (default)
        result = encoder.encode(long_seq, chunk_size=1000, overlap=100)
        assert result.shape[0] == 1  # aggregated to single embedding

        # Without aggregation
        result_chunks = encoder.encode(
            long_seq, chunk_size=1000, overlap=100, aggregate=None
        )
        assert result_chunks.shape[0] > 1  # multiple chunks

    def test_predict_returns_dict(self):
        """predict() should return dict of tensors."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cpu")
        result = encoder.predict("ATGC" * 100)

        assert isinstance(result, dict)
        assert len(result) > 0
        for name, tensor in result.items():
            assert isinstance(name, str)
            assert isinstance(tensor, torch.Tensor)

    def test_modality_is_dna(self):
        """modality property should return 'dna'."""
        from manylatents.dogma.encoders import AlphaGenomeEncoder

        encoder = AlphaGenomeEncoder(device="cpu")
        assert encoder.modality == "dna"


def test_import_from_encoders():
    """AlphaGenomeEncoder should be importable from encoders module."""
    from manylatents.dogma.encoders import AlphaGenomeEncoder

    assert AlphaGenomeEncoder is not None
