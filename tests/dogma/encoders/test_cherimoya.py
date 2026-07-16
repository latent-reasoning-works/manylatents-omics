"""Tests for CherimoyaEncoder.

Unit tests only — they construct the encoder and check its static contract
without loading a model, so they run without ``cherimoya``/``tangermeme``/a GPU
installed (heavy imports are lazy inside ``_load_model``/``_one_hot``). Mirrors
the AlphaGenome encoder tests.
"""

import pytest


class TestCherimoyaEncoderUnit:
    """Unit tests for CherimoyaEncoder (no model loading)."""

    def test_modality_is_dna(self):
        from manylatents.dogma.encoders import CherimoyaEncoder

        assert CherimoyaEncoder(device="cpu").modality == "dna"

    def test_embedding_dim_is_n_tracks(self):
        from manylatents.dogma.encoders import CherimoyaEncoder

        assert CherimoyaEncoder(device="cpu").embedding_dim == 1
        assert CherimoyaEncoder(n_tracks=4, device="cpu").embedding_dim == 4

    def test_context_length_defaults_to_window(self):
        from manylatents.dogma.encoders import CherimoyaEncoder

        assert CherimoyaEncoder(device="cpu").context_length == 2114
        assert CherimoyaEncoder(window=1000, device="cpu").context_length == 1000

    def test_invalid_n_layers(self):
        from manylatents.dogma.encoders import CherimoyaEncoder

        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            CherimoyaEncoder(n_layers=0, device="cpu")

    def test_invalid_n_tracks(self):
        from manylatents.dogma.encoders import CherimoyaEncoder

        with pytest.raises(ValueError, match="n_tracks must be >= 1"):
            CherimoyaEncoder(n_tracks=0, device="cpu")

    def test_split_sequence(self):
        from manylatents.dogma.encoders import CherimoyaEncoder

        seq = "ATGC" * 100  # 400 bp
        chunks = CherimoyaEncoder._split_sequence(seq, chunk_size=100, overlap=20)
        assert len(chunks) == 5  # 400 / (100-20)
        assert all(len(c) <= 100 for c in chunks)

    def test_split_sequence_overlap_error(self):
        from manylatents.dogma.encoders import CherimoyaEncoder

        with pytest.raises(ValueError, match="overlap .* must be < chunk_size"):
            CherimoyaEncoder._split_sequence("ATGC", chunk_size=100, overlap=100)

    def test_checkpoint_stored(self):
        from manylatents.dogma.encoders import CherimoyaEncoder

        enc = CherimoyaEncoder(checkpoint="dnase_k562.torch", device="cpu")
        assert enc.checkpoint == "dnase_k562.torch"
        assert enc._model is None  # lazy — not loaded on construction


def test_import_from_encoders():
    """CherimoyaEncoder should be importable from the encoders module + __all__."""
    from manylatents.dogma import encoders

    assert hasattr(encoders, "CherimoyaEncoder")
    assert "CherimoyaEncoder" in encoders.__all__


class TestCherimoyaTrainPort:
    """The training port — CLI command assembly + guard (no cherimoya run)."""

    def _build(self, **kw):
        from pathlib import Path

        from manylatents.dogma.encoders.cherimoya import _build_pipeline_commands

        defaults = dict(
            genome=Path("hg38.fa"), signal=Path("k562_dnase.bigWig"),
            name="k562_dnase", out_dir=Path("/out"), motifs=Path("JASPAR_2024.meme"),
        )
        defaults.update(kw)
        return _build_pipeline_commands(**defaults)

    def test_json_and_run_commands(self):
        json_cmd, run_cmd, ckpt = self._build()
        assert json_cmd[:2] == ["cherimoya", "pipeline-json"]
        assert json_cmd[json_cmd.index("-i") + 1] == "k562_dnase.bigWig"
        assert json_cmd[json_cmd.index("-n") + 1] == "k562_dnase"
        # run cmd points at the json that pipeline-json wrote
        assert run_cmd[:2] == ["cherimoya", "pipeline"]
        assert run_cmd[run_cmd.index("-p") + 1] == json_cmd[json_cmd.index("-o") + 1]
        assert str(ckpt).endswith("k562_dnase.torch")

    def test_unstranded_default_and_toggle(self):
        assert "-u" in self._build()[0]
        assert "-u" not in self._build(unstranded=False)[0]

    def test_peaks_and_control_are_optional(self):
        from pathlib import Path

        assert "-p" not in self._build()[0] and "-c" not in self._build()[0]
        json_cmd = self._build(peaks=Path("p.narrowPeak"), control=Path("in.bigWig"))[0]
        assert json_cmd[json_cmd.index("-p") + 1] == "p.narrowPeak"
        assert json_cmd[json_cmd.index("-c") + 1] == "in.bigWig"

    def test_train_requires_cherimoya_cli(self, monkeypatch):
        """With no `cherimoya` on PATH, train() fails fast (never half-runs)."""
        import shutil

        from manylatents.dogma.encoders import CherimoyaEncoder

        monkeypatch.setattr(shutil, "which", lambda _: None)
        with pytest.raises(RuntimeError, match="cherimoya.*not on PATH"):
            CherimoyaEncoder.train(
                genome="hg38.fa", signal="s.bigWig", name="n",
                out_dir="/tmp/does-not-run", motifs="m.meme",
            )
