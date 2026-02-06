"""AlphaGenome regulatory effect predictor.

Predicts regulatory effects of variants using AlphaGenome foundation model.
Follows the LatentModule pattern but outputs per-track effect scores instead
of embeddings.

Usage:
    >>> predictor = AlphaGenomePredictor(
    ...     ontology="EFO:0002067",  # K562
    ...     cell_type="K562",
    ...     datamodule=variant_datamodule,
    ... )
    >>> effects = predictor.fit_transform(dummy_tensor)

Effect scores computed as:
    L2_norm(MUT_prediction - WT_prediction)

Output tracks:
    - splice: Splice site predictions (donor + acceptor)
    - rna_seq: RNA-seq expression
    - cage: CAGE-seq TSS activity
    - atac: ATAC-seq accessibility
    - dnase: DNase-seq open chromatin
    - aggregate: Mean across all tracks
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


class AlphaGenomePredictor(LatentModule):
    """Predict regulatory effects using AlphaGenome.

    Takes paired WT/MUT sequences from VariantDataModule and computes
    per-track effect scores.

    Args:
        ontology: Ontology term for cell type. Options:
            - K562: EFO:0002067 (chronic myelogenous leukemia)
            - HepG2: EFO:0001187 (hepatocellular carcinoma)
            - GM12878: EFO:0002784 (lymphoblastoid)
            - generic: UBERON:0000468 (multicellular organism)
        cell_type: Cell type label for output directory (K562, HepG2, GM12878).
        save_path: Path to save results (.tsv and .npy files).
        target_length: Sequence length for AlphaGenome (must be multiple of 128).
            Default 16384.
        **kwargs: Passed to LatentModule (datamodule, init_seed, etc.)

    Output:
        DataFrame with columns: variation_id, label, effect_{track} for each track.

    GPU Requirements:
        Requires Ampere+ GPU (A100, L40S, H100) for BF16 support.
    """

    TRACK_NAMES = ["splice", "rna_seq", "cage", "atac", "dnase"]
    DEFAULT_ONTOLOGY = "UBERON:0000468"

    ONTOLOGY_MAP = {
        "K562": "EFO:0002067",
        "HepG2": "EFO:0001187",
        "GM12878": "EFO:0002784",
        "generic": "UBERON:0000468",
    }

    def __init__(
        self,
        ontology: Optional[str] = None,
        cell_type: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        target_length: int = 16384,
        n_components: int = 6,  # 5 tracks + aggregate
        **kwargs,
    ):
        super().__init__(n_components=n_components, **kwargs)

        # Resolve ontology from cell_type if provided
        if cell_type and ontology is None:
            ontology = self.ONTOLOGY_MAP.get(cell_type, self.DEFAULT_ONTOLOGY)

        self._ontology = ontology or self.DEFAULT_ONTOLOGY
        self._cell_type = cell_type
        self._save_path = Path(save_path) if save_path else None
        self._target_length = target_length

        # Lazy-loaded model
        self._model = None

    def _ensure_model_loaded(self) -> None:
        """Lazy-load AlphaGenome model."""
        if self._model is None:
            logger.info("Loading AlphaGenome model from HuggingFace...")
            from alphagenome_research.model import dna_model
            self._model = dna_model.create_from_huggingface("all_folds")
            logger.info("AlphaGenome model loaded successfully!")

    def _prepare_sequence(self, seq: str) -> str:
        """Prepare sequence to exact target length for AlphaGenome."""
        if len(seq) > self._target_length:
            # Trim from center
            trim = (len(seq) - self._target_length) // 2
            return seq[trim:trim + self._target_length]
        elif len(seq) < self._target_length:
            # Pad with N
            pad = (self._target_length - len(seq)) // 2
            return "N" * pad + seq + "N" * (self._target_length - len(seq) - pad)
        return seq

    def _inject_variant(self, seq: str, center: int, ref: str, alt: str) -> str:
        """Inject variant at center position."""
        actual_ref = seq[center:center + len(ref)]
        if actual_ref.upper() != ref.upper():
            # Try to find ref nearby (within 5bp)
            for offset in range(-5, 6):
                check_pos = center + offset
                if 0 <= check_pos < len(seq) - len(ref):
                    if seq[check_pos:check_pos + len(ref)].upper() == ref.upper():
                        center = check_pos
                        break
        return seq[:center] + alt + seq[center + len(ref):]

    def _predict_sequence(self, wt_seq: str, mut_seq: str) -> Dict[str, float]:
        """Run AlphaGenome on WT and MUT sequences, return effect scores."""
        from alphagenome_research.model import dna_model

        # Prepare sequences to exact length
        wt_seq = self._prepare_sequence(wt_seq)
        mut_seq = self._prepare_sequence(mut_seq)

        # Request all track outputs
        output_types = [
            dna_model.OutputType.SPLICE_SITES,
            dna_model.OutputType.RNA_SEQ,
            dna_model.OutputType.CAGE,
            dna_model.OutputType.ATAC,
            dna_model.OutputType.DNASE,
        ]

        ontology_terms = [self._ontology]

        # Predict on WT and MUT
        wt_output = self._model.predict_sequence(
            sequence=wt_seq,
            requested_outputs=output_types,
            ontology_terms=ontology_terms,
        )
        mut_output = self._model.predict_sequence(
            sequence=mut_seq,
            requested_outputs=output_types,
            ontology_terms=ontology_terms,
        )

        # Extract effect scores (L2 norm of difference)
        effects = {}
        track_attr_mapping = {
            "splice": "splice_sites",
            "rna_seq": "rna_seq",
            "cage": "cage",
            "atac": "atac",
            "dnase": "dnase",
        }

        for track_name, attr_name in track_attr_mapping.items():
            try:
                wt_data = getattr(wt_output, attr_name, None)
                mut_data = getattr(mut_output, attr_name, None)

                if wt_data is not None and mut_data is not None:
                    wt_vals = np.array(wt_data.values if hasattr(wt_data, "values") else wt_data)
                    mut_vals = np.array(mut_data.values if hasattr(mut_data, "values") else mut_data)
                    diff = mut_vals - wt_vals
                    effects[f"effect_{track_name}"] = float(np.linalg.norm(diff.flatten()))
                else:
                    effects[f"effect_{track_name}"] = 0.0
            except Exception as e:
                logger.warning(f"Error extracting {track_name}: {e}")
                effects[f"effect_{track_name}"] = 0.0

        # Aggregate across all tracks
        track_effects = [v for k, v in effects.items() if k.startswith("effect_") and v > 0]
        effects["effect_aggregate"] = float(np.mean(track_effects)) if track_effects else 0.0

        return effects

    def fit(self, x: Tensor, y: Tensor = None) -> None:
        """No-op fit - AlphaGenome is pretrained."""
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Run AlphaGenome on all variants from datamodule.

        Args:
            x: Input tensor (ignored - sequences come from datamodule).

        Returns:
            Tensor of shape (N, 6) with effect scores for each track + aggregate.
        """
        self._ensure_model_loaded()

        if self.datamodule is None:
            raise ValueError("AlphaGenomePredictor requires a datamodule")

        # Get paired sequences
        pairs = self.datamodule.get_sequence_pairs()
        wt_seqs = pairs["wt"]
        mut_seqs = pairs["mut"]
        variant_ids = self.datamodule.get_variant_ids()
        labels = self.datamodule.get_labels()

        logger.info(f"Running AlphaGenome on {len(wt_seqs)} variants...")
        logger.info(f"Cell type: {self._cell_type}, Ontology: {self._ontology}")

        results = []
        n_errors = 0

        for i, (wt_seq, mut_seq, var_id, label) in enumerate(
            tqdm(zip(wt_seqs, mut_seqs, variant_ids, labels), total=len(wt_seqs), desc="AlphaGenome")
        ):
            try:
                effects = self._predict_sequence(wt_seq, mut_seq)
                effects["variation_id"] = var_id
                effects["label"] = label
                results.append(effects)
            except Exception as e:
                logger.warning(f"Error on {var_id}: {e}")
                n_errors += 1
                continue

        logger.info(f"Completed: {len(results)} successful, {n_errors} errors")

        if len(results) == 0:
            raise ValueError("No valid results computed")

        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        if self._save_path:
            self._save_results(df)

        # Return effect scores as tensor (N, 6)
        effect_cols = [f"effect_{t}" for t in self.TRACK_NAMES] + ["effect_aggregate"]
        effects_tensor = torch.tensor(df[effect_cols].values, dtype=torch.float32)

        return effects_tensor

    def _save_results(self, df: pd.DataFrame) -> None:
        """Save results to TSV and NPY files."""
        self._save_path.parent.mkdir(parents=True, exist_ok=True)

        # Get variant type from datamodule
        variant_type = getattr(self.datamodule, "variant_type", "unknown")

        # Save TSV
        tsv_path = self._save_path / f"{variant_type}_effects.tsv"
        df.to_csv(tsv_path, sep="\t", index=False)
        logger.info(f"Saved {len(df)} results to {tsv_path}")

        # Save NPY per track
        for track in self.TRACK_NAMES + ["aggregate"]:
            col = f"effect_{track}"
            if col in df.columns:
                npy_path = self._save_path / f"{variant_type}_effect_{track}.npy"
                np.save(npy_path, df[col].values)

        # Save labels
        np.save(self._save_path / f"{variant_type}_labels.npy", df["label"].values)

        # Print summary
        logger.info(f"\n=== Effect Score Summary for {variant_type} ===")
        for track in self.TRACK_NAMES + ["aggregate"]:
            col = f"effect_{track}"
            if col in df.columns:
                vals = df[col]
                logger.info(f"{track}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    def cleanup(self) -> None:
        """Release model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"cell_type={self._cell_type}, "
            f"ontology={self._ontology}, "
            f"target_length={self._target_length})"
        )
