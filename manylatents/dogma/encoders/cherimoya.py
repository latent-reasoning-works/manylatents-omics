"""Cherimoya encoder — the tiny regulatory DNA S2F model.

Cherimoya (Schreiber, Programmable Genomics Laboratory; `jmschrei/cherimoya`,
MIT) is a compact ChromBPNet-successor that predicts genomic coverage tracks
(TF binding, ATAC/DNase accessibility, transcription initiation, ...) from DNA
sequence. The default 9-layer model is only ~610K parameters and forwards in
under a millisecond per batch, with a pure-PyTorch CPU fallback. Within the
central-dogma stack it is the *small expert* for the regulatory (DNA) arm — a
lightweight counterpart to :class:`AlphaGenomeEncoder`.

Unlike AlphaGenome (one pretrained all-tracks model), Cherimoya is trained per
experiment, so pass a trained ``checkpoint`` (a ``Cherimoya.load``-able
``.torch`` bundle). With no checkpoint the encoder instantiates a fresh
(untrained) model of the requested shape — useful for shape/plumbing tests only.

Predict-first: Cherimoya's native output is ``(profile, log_counts)`` per track,
exposed via :meth:`predict`. :meth:`encode` returns the per-track log-count head
as a compact functional summary vector (this is Cherimoya's natural embedding —
it is not a learned latent like ESM/Evo2).

References:
    - GitHub: https://github.com/jmschrei/cherimoya
    - PyPI:   https://pypi.org/project/cherimoya/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from manylatents.algorithms.latent.foundation_encoder import FoundationEncoder

logger = logging.getLogger(__name__)

# Cherimoya ships NO canonical/pretrained weights in this branch (it is trained
# per experiment). Every inference path emits one of these so no run can be
# mistaken for a validated model. See CHERIMOYA_INTEGRATION / CLAUDE notes.
_NO_WEIGHTS_UNTRAINED = (
    "Cherimoya: NO CANONICAL WEIGHTS. No checkpoint given -> running a FRESH "
    "UNTRAINED model; predictions are MEANINGLESS (shape/plumbing only)."
)
_NO_WEIGHTS_CHECKPOINT = (
    "Cherimoya: NO CANONICAL WEIGHTS ship in this branch. Running inference with "
    "user-supplied checkpoint %s -- outputs are provisional (checkpoint provenance "
    "unvalidated), not validated model predictions."
)


def _build_pipeline_commands(
    genome, signal, name: str, out_dir: Path, motifs,
    peaks=None, control=None, unstranded: bool = True,
) -> Tuple[List[str], List[str], Path]:
    """Assemble the ``cherimoya`` CLI training commands + checkpoint path.

    Faithful to the verified v0.2.0 CLI (``pipeline-json`` then ``pipeline``);
    pure so it is unit-testable without invoking cherimoya. Returns
    ``(json_cmd, run_cmd, checkpoint_path)``. ``signal`` may be bigWig (BAM→bigWig
    is auto-skipped) or BAM; ``unstranded`` adds ``-u``.
    """
    json_path = Path(out_dir) / f"{name}.pipeline.json"
    json_cmd = [
        "cherimoya", "pipeline-json",
        "-s", str(genome), "-i", str(signal),
        "-m", str(motifs), "-n", name, "-o", str(json_path),
    ]
    if peaks is not None:
        json_cmd += ["-p", str(peaks)]
    if control is not None:
        json_cmd += ["-c", str(control)]
    if unstranded:
        json_cmd += ["-u"]
    run_cmd = ["cherimoya", "pipeline", "-p", str(json_path)]
    return json_cmd, run_cmd, Path(out_dir) / f"{name}.torch"


class CherimoyaEncoder(FoundationEncoder):
    """Cherimoya encoder for regulatory DNA sequences.

    Args:
        checkpoint: Path to a trained Cherimoya ``.torch`` bundle. If ``None``,
            a fresh (untrained) model of shape ``(n_filters, n_layers)`` is built
            — for plumbing/shape checks only, not meaningful predictions.
        n_filters: Channel width (used only when ``checkpoint is None``).
        n_layers: Number of Cheri Blocks (used only when ``checkpoint is None``).
        n_tracks: Number of output tracks the checkpoint predicts; sets
            :attr:`embedding_dim` (the log-count summary width). Default 1
            (single-task). Multi-task checkpoints expose several tracks at once.
        window: Input receptive-field window in bp (Cherimoya default 2114).
        device: ``"cuda"`` for the inference megakernel; ``"cpu"`` uses the
            pure-PyTorch fallback.

    Example:
        >>> enc = CherimoyaEncoder(checkpoint="dnase_k562.torch", device="cpu")
        >>> tracks = enc.predict("ACGT" * 528 + "AA")   # {'profile', 'log_counts'}
        >>> summary = enc.encode("ACGT" * 528 + "AA")   # (1, n_tracks) log-counts
    """

    # Cherimoya's default receptive-field window: 2114 bp in -> 1000 bp out.
    DEFAULT_WINDOW = 2114

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        n_filters: int = 128,
        n_layers: int = 9,
        n_tracks: int = 1,
        window: int = DEFAULT_WINDOW,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if n_tracks < 1:
            raise ValueError(f"n_tracks must be >= 1, got {n_tracks}")

        self.checkpoint = checkpoint
        self._n_filters = n_filters
        self._n_layers = n_layers
        self._n_tracks = n_tracks
        self._window = window
        self._counter = None  # LogCountWrapper, set on load

    def _load_model(self) -> None:
        """Lazy-load Cherimoya from a checkpoint (or build a fresh model)."""
        try:
            from cherimoya import Cherimoya, LogCountWrapper
        except ImportError as e:
            raise ImportError(
                "Cherimoya requires the 'cherimoya' package. "
                "Install with: uv sync --extra cherimoya  (or: pip install cherimoya)"
            ) from e

        if self.checkpoint is not None:
            logger.warning(_NO_WEIGHTS_CHECKPOINT, self.checkpoint)
            model = Cherimoya.load(str(self.checkpoint), device=self.device)
        else:
            logger.warning(_NO_WEIGHTS_UNTRAINED)
            model = Cherimoya(
                n_filters=self._n_filters, n_layers=self._n_layers
            ).to(self.device)

        self._model = model.eval()
        # LogCountWrapper exposes a single (N, n_tracks) log-count tensor.
        self._counter = LogCountWrapper(self._model)

    @classmethod
    def train(
        cls,
        genome,
        signal,
        name: str,
        out_dir,
        motifs,
        *,
        peaks=None,
        control=None,
        unstranded: bool = True,
        device: str = "cpu",
        n_tracks: int = 1,
        check: bool = True,
        **encoder_kwargs,
    ) -> "CherimoyaEncoder":
        """Train a Cherimoya checkpoint, then return an encoder loaded with it.

        The *training* port: Cherimoya ships no weights and none is open-source,
        so a checkpoint must be trained per track. This is a thin driver over the
        ``cherimoya`` CLI (``pipeline-json`` -> ``pipeline``, v0.2.0) — the training
        loop (Muon/EMA/Kendall-Gal) stays owned by upstream. Requires the
        ``cherimoya`` CLI on PATH; does not pick tracks/data (Path A / ENCODE: see
        dogma-signal-manifold#38). ``signal`` may be bigWig or BAM.
        """
        import shutil
        import subprocess

        if shutil.which("cherimoya") is None:
            raise RuntimeError(
                "The 'cherimoya' CLI is not on PATH. Install: pip install cherimoya"
            )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_cmd, run_cmd, ckpt = _build_pipeline_commands(
            genome, signal, name, out_dir, motifs, peaks, control, unstranded
        )
        logger.info("cherimoya pipeline-json: %s", " ".join(json_cmd))
        subprocess.run(json_cmd, check=check)
        logger.info("cherimoya pipeline (fit): %s", " ".join(run_cmd))
        subprocess.run(run_cmd, check=check)
        logger.warning(
            "Trained a NEW Cherimoya checkpoint at %s. NO CANONICAL WEIGHTS exist; "
            "its provenance is exactly this training run on '%s' -- validate before "
            "trusting its scores.",
            ckpt, signal,
        )
        return cls(checkpoint=str(ckpt), device=device, n_tracks=n_tracks, **encoder_kwargs)

    def _one_hot(self, sequence: str) -> Tensor:
        """One-hot encode DNA to ``(1, 4, L)`` on the encoder device.

        Uses tangermeme (Cherimoya's own IO dependency); unknown bases (``N``)
        become an all-zero column.
        """
        from tangermeme.utils import one_hot_encode

        x = one_hot_encode(sequence).unsqueeze(0).float()
        return x.to(self.device)

    def predict(self, sequence: str) -> Dict[str, Tensor]:
        """Predict regulatory tracks for a DNA sequence.

        Returns a dict with:
            - ``profile``: base-resolution profile logits, ``(1, n_tracks, out_len)``
            - ``log_counts``: per-track total log-counts, ``(1, n_tracks)``

        Variant effect (as used by the signal-manifold scorer) is the shift in
        ``log_counts`` between ALT and REF windows.
        """
        self._ensure_loaded()
        x = self._one_hot(sequence)
        with torch.no_grad():
            profile, log_counts = self._model(x)
        return {"profile": profile, "log_counts": log_counts}

    def encode(self, sequence: str) -> Tensor:
        """Compact functional summary: the per-track log-count head ``(1, n_tracks)``.

        Cherimoya is a predict-first model; this is its natural embedding, not a
        learned latent. For the full track output use :meth:`predict`.
        """
        self._ensure_loaded()
        x = self._one_hot(sequence)
        with torch.no_grad():
            log_counts = self._counter(x)
        return log_counts.reshape(1, -1)

    @staticmethod
    def _split_sequence(sequence: str, chunk_size: int, overlap: int) -> List[str]:
        """Split a long sequence into overlapping chunks (windowing helper)."""
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
        step = chunk_size - overlap
        return [
            sequence[i : i + chunk_size]
            for i in range(0, len(sequence), step)
            if sequence[i : i + chunk_size]
        ]

    @property
    def modality(self) -> str:
        return "dna"

    @property
    def embedding_dim(self) -> int:
        """Width of the :meth:`encode` summary — the number of output tracks."""
        return self._n_tracks

    @property
    def context_length(self) -> int:
        """Input receptive-field window in bp (Cherimoya default 2114)."""
        return self._window
