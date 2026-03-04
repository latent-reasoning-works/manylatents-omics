"""AlphaMissense pathogenicity scorer.

Looks up pre-computed AlphaMissense pathogenicity scores for missense variants.
Data source: Cheng et al. (Science 2023), available from Zenodo.

Usage:
    >>> scorer = AlphaMissenseScorer("path/to/AlphaMissense_hg38.tsv.gz")
    >>> scores = scorer.score(variants_df)  # DataFrame with CHROM, POS, REF, ALT
    >>> # scores: np.ndarray of floats, NaN for variants not in AlphaMissense

The TSV is loaded lazily on first call to score(). For the full genome file
(~71M rows), expect ~4GB RAM and ~30s load time.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# AlphaMissense hg38 TSV columns (after the 4-line header comment block)
_GENOMIC_COLS = ["CHROM", "POS", "REF", "ALT"]
_PROTEIN_COLS = ["uniprot_id", "protein_variant"]
_SCORE_COL = "am_pathogenicity"
_CLASS_COL = "am_class"


class AlphaMissenseScorer:
    """Look up pre-computed AlphaMissense pathogenicity scores.

    Supports two matching modes:
        - "genomic": match by (CHROM, POS, REF, ALT) in hg38 coordinates
        - "protein": match by (uniprot_id, protein_variant) e.g. ("P38398", "V1736A")

    Args:
        predictions_path: Path to AlphaMissense_hg38.tsv.gz (genomic mode)
            or AlphaMissense_aa_substitutions.tsv.gz (protein mode).
        match_by: Matching strategy. "genomic" or "protein".
    """

    def __init__(
        self,
        predictions_path: Union[str, Path],
        match_by: str = "genomic",
    ):
        if match_by not in ("genomic", "protein"):
            raise ValueError(f"match_by must be 'genomic' or 'protein', got '{match_by}'")
        self._path = Path(predictions_path)
        self._match_by = match_by
        self._lookup: Optional[dict] = None

    @property
    def name(self) -> str:
        return "alphamissense"

    def _load(self) -> None:
        """Load the TSV into a dict keyed for O(1) lookup."""
        if self._lookup is not None:
            return

        logger.info("Loading AlphaMissense predictions from %s ...", self._path)

        # AlphaMissense hg38 files have comment lines (starting with "# ")
        # followed by a header line starting with "#CHROM". We skip the
        # pure comment lines and let pandas read "#CHROM..." as the header.
        opener = gzip.open if self._path.suffix == ".gz" else open
        with opener(self._path, "rt") as f:
            skip = 0
            for line in f:
                if line.startswith("# ") or line.strip() == "#":
                    skip += 1
                else:
                    break  # header line (e.g. "#CHROM\t...") or data

        if self._match_by == "genomic":
            df = pd.read_csv(
                self._path,
                sep="\t",
                skiprows=skip,
                usecols=["#CHROM", "POS", "REF", "ALT", _SCORE_COL, _CLASS_COL],
                dtype={"#CHROM": str, "POS": np.int64, "REF": str, "ALT": str,
                       _SCORE_COL: np.float32, _CLASS_COL: str},
            )
            df.rename(columns={"#CHROM": "CHROM"}, inplace=True)
            # Strip "chr" prefix if present for flexible matching
            df["CHROM"] = df["CHROM"].str.replace("^chr", "", regex=True)
            # Key: "CHROM:POS:REF:ALT"
            keys = (
                df["CHROM"] + ":" +
                df["POS"].astype(str) + ":" +
                df["REF"] + ":" +
                df["ALT"]
            )
        else:
            # Protein file: comment lines start with "# ", header is
            # "uniprot_id\tprotein_variant\t..."
            df = pd.read_csv(
                self._path,
                sep="\t",
                skiprows=skip,
                usecols=["uniprot_id", "protein_variant", _SCORE_COL, _CLASS_COL],
                dtype={"uniprot_id": str, "protein_variant": str,
                       _SCORE_COL: np.float32, _CLASS_COL: str},
            )
            # Key: "uniprot_id:protein_variant"
            keys = df["uniprot_id"] + ":" + df["protein_variant"]

        self._lookup = dict(zip(keys, df[_SCORE_COL].values))
        logger.info("Loaded %d AlphaMissense predictions.", len(self._lookup))

    def _make_key(self, row: pd.Series) -> str:
        """Build lookup key from a variant row."""
        if self._match_by == "genomic":
            chrom = str(row.get("CHROM", row.get("chromosome", "")))
            chrom = chrom.replace("chr", "")
            pos = row.get("POS", row.get("start", ""))
            ref = row.get("REF", row.get("ref", ""))
            alt = row.get("ALT", row.get("alt", ""))
            return f"{chrom}:{pos}:{ref}:{alt}"
        else:
            uid = row.get("uniprot_id", "")
            pvar = row.get("protein_variant", "")
            return f"{uid}:{pvar}"

    def score(self, variants: pd.DataFrame) -> np.ndarray:
        """Score variants by looking up AlphaMissense predictions.

        Args:
            variants: DataFrame with columns matching the match_by mode:
                - genomic: needs CHROM/chromosome, POS/start, REF/ref, ALT/alt
                - protein: needs uniprot_id, protein_variant

        Returns:
            Array of pathogenicity scores (float32). NaN for variants not found
            in AlphaMissense.
        """
        self._load()
        scores = np.full(len(variants), np.nan, dtype=np.float32)
        for i, (_, row) in enumerate(variants.iterrows()):
            key = self._make_key(row)
            val = self._lookup.get(key)
            if val is not None:
                scores[i] = val
        return scores

    def score_from_keys(
        self,
        chroms: list[str],
        positions: list[int],
        refs: list[str],
        alts: list[str],
    ) -> np.ndarray:
        """Score variants from parallel arrays (avoids DataFrame overhead).

        Only works in genomic mode. For large variant sets, this is faster
        than constructing a DataFrame.
        """
        if self._match_by != "genomic":
            raise ValueError("score_from_keys only works with match_by='genomic'")
        self._load()
        scores = np.full(len(chroms), np.nan, dtype=np.float32)
        for i, (c, p, r, a) in enumerate(zip(chroms, positions, refs, alts)):
            key = f"{str(c).replace('chr', '')}:{p}:{r}:{a}"
            val = self._lookup.get(key)
            if val is not None:
                scores[i] = val
        return scores

    def __repr__(self) -> str:
        loaded = len(self._lookup) if self._lookup else "not loaded"
        return f"AlphaMissenseScorer(path='{self._path}', match_by='{self._match_by}', entries={loaded})"
