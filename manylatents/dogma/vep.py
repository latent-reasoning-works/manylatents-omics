"""Variant-effect prediction utilities for protein language models.

Pure-Python helpers for mutation parsing, sequence manipulation, and
variant-effect metrics (delta-norm, cosine-distance, LLR, LID). Designed
to compose with ``manylatents.dogma.encoders.ESMEncoder.encode_with_logits``
but duck-typed: any encoder that exposes ``encode_with_logits(seq) ->
(embedding, logits)``, ``tok_id(aa) -> int``, and ``max_length: int``
works.

Ported from ``lrw-vep-ub2026/experiments/notebooks/vep_utils.py`` as part
of the 2.11 collapse — both the workshop notebook's Path A and the
canonical Phase-1 Path B/C now share this module. The HF-transformers
``ESM1bEncoder`` in ``vep_utils.py`` is preserved there as a Colab
convenience (HF transformers ships pre-installed on Colab; fair-esm does
not); the *helpers* below are the single source of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

AA_FULL_NAMES = {
    "A": "Alanine", "C": "Cysteine", "D": "Aspartate", "E": "Glutamate",
    "F": "Phenylalanine", "G": "Glycine", "H": "Histidine", "I": "Isoleucine",
    "K": "Lysine", "L": "Leucine", "M": "Methionine", "N": "Asparagine",
    "P": "Proline", "Q": "Glutamine", "R": "Arginine", "S": "Serine",
    "T": "Threonine", "V": "Valine", "W": "Tryptophan", "Y": "Tyrosine",
}


@dataclass(frozen=True)
class MutationSpec:
    """A single point mutation."""
    wt_aa: str
    position: int  # 1-indexed
    mut_aa: str

    def __str__(self) -> str:
        return f"{self.wt_aa}{self.position}{self.mut_aa}"


def validate_sequence(sequence: str, max_length: int = 1024) -> str:
    """Validate a protein sequence. Returns the (possibly truncated) sequence.

    Raises ValueError for empty sequences or invalid characters. Prints a
    warning and truncates if longer than ``max_length``.
    """
    if not sequence:
        raise ValueError("Sequence is empty")
    if sequence.lstrip().startswith(">"):
        raise ValueError(
            "Input looks like FASTA (starts with '>'). "
            "Please paste only the sequence, without the header line."
        )
    sequence = "".join(sequence.split()).upper()
    for i, aa in enumerate(sequence):
        if aa not in AA_ALPHABET:
            raise ValueError(
                f"Invalid amino acid {aa!r} at position {i + 1}. "
                f"Valid residues: {AA_ALPHABET}"
            )
    if len(sequence) > max_length:
        print(
            f"Warning: Sequence too long ({len(sequence)} AA). "
            f"ESM-1b max context is {max_length}. Truncating to first {max_length}."
        )
        sequence = sequence[:max_length]
    return sequence


def parse_mutation(mutation_str: str) -> MutationSpec:
    """Parse a mutation string like 'A23T' into a MutationSpec."""
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation_str.strip().upper())
    if not match:
        raise ValueError(
            f"Could not parse mutation {mutation_str!r}. "
            f"Expected format: 'A23T' (wt_aa, position, mut_aa)."
        )
    wt_aa, pos_str, mut_aa = match.groups()
    if wt_aa not in AA_ALPHABET:
        raise ValueError(f"Invalid amino acid {wt_aa!r} in mutation {mutation_str!r}")
    if mut_aa not in AA_ALPHABET:
        raise ValueError(f"Invalid amino acid {mut_aa!r} in mutation {mutation_str!r}")
    return MutationSpec(wt_aa=wt_aa, position=int(pos_str), mut_aa=mut_aa)


def validate_mutation(sequence: str, mutation: MutationSpec) -> None:
    """Verify a mutation is consistent with its WT sequence."""
    if mutation.position < 1 or mutation.position > len(sequence):
        raise ValueError(
            f"Position {mutation.position} out of range "
            f"(sequence length: {len(sequence)})"
        )
    actual_aa = sequence[mutation.position - 1]
    if actual_aa != mutation.wt_aa:
        actual_name = AA_FULL_NAMES.get(actual_aa, actual_aa)
        raise ValueError(
            f"Position {mutation.position} is {actual_name} ({actual_aa}), "
            f"not {mutation.wt_aa}. Did you mean "
            f"{actual_aa}{mutation.position}{mutation.mut_aa}?"
        )
    if mutation.wt_aa == mutation.mut_aa:
        raise ValueError(
            f"WT and MUT residues are identical ({mutation.wt_aa}) — "
            f"this is not a mutation"
        )


def apply_mutation(sequence: str, mutation: MutationSpec) -> str:
    """Apply a mutation to a sequence; return the mutant sequence."""
    validate_mutation(sequence, mutation)
    return (
        sequence[: mutation.position - 1]
        + mutation.mut_aa
        + sequence[mutation.position:]
    )


def truncate_around_mutation(
    sequence: str, position_1idx: int, window: int = 1024,
) -> tuple[str, int]:
    """Center-window a sequence around a mutation position.

    Returns (truncated_seq, new_position_1idx). For sequences shorter
    than ``window``, returns the original sequence and position unchanged.
    The mutation position is preserved relative to the returned sequence.
    """
    if len(sequence) <= window:
        return sequence, position_1idx
    half = window // 2
    start = max(0, position_1idx - 1 - half)
    end = start + window
    if end > len(sequence):
        end = len(sequence)
        start = end - window
    new_pos = position_1idx - start
    return sequence[start:end], new_pos


def encode_variant(encoder, sequence: str, mutation_str: str) -> dict:
    """Encode WT and MUT for a single point mutation.

    The encoder is duck-typed: it must provide
    ``encode_with_logits(seq) -> (embedding, logits)`` and a
    ``max_length`` attribute. ``manylatents.dogma.encoders.ESMEncoder``
    satisfies both.

    Returns a dict with:
        wt_sequence, mut_sequence: str
        wt_embedding, mut_embedding: ``np.ndarray`` (shape: D,)
        wt_logits, mut_logits: ``np.ndarray`` (shape: L+2, vocab)
        mutation: MutationSpec
    """
    sequence = validate_sequence(sequence, max_length=encoder.max_length)
    mutation = parse_mutation(mutation_str)
    mut_sequence = apply_mutation(sequence, mutation)

    wt_emb, wt_logits = encoder.encode_with_logits(sequence)
    mut_emb, mut_logits = encoder.encode_with_logits(mut_sequence)

    return {
        "wt_sequence": sequence,
        "mut_sequence": mut_sequence,
        "wt_embedding": wt_emb,
        "mut_embedding": mut_emb,
        "wt_logits": wt_logits,
        "mut_logits": mut_logits,
        "mutation": mutation,
    }


def compute_delta_norm(
    wt_embedding: np.ndarray, mut_embedding: np.ndarray,
) -> float:
    """L2 norm of ``mut - wt``."""
    return float(np.linalg.norm(mut_embedding - wt_embedding))


def compute_cosine_distance(
    wt_embedding: np.ndarray, mut_embedding: np.ndarray,
) -> float:
    """Cosine distance = 1 - cosine similarity. Returns NaN if either norm is 0."""
    dot = float(np.dot(wt_embedding, mut_embedding))
    norms = float(np.linalg.norm(wt_embedding) * np.linalg.norm(mut_embedding))
    if norms < 1e-10:
        return float("nan")
    return 1.0 - dot / norms


def compute_llr(
    wt_logits: np.ndarray,
    mutation: MutationSpec,
    wt_token_id: int,
    mut_token_id: int,
) -> float:
    """ESM-1b masked-LM LLR per Brandes et al. 2023 Methods.

        LLR = log P(mut_aa | WT_seq) - log P(wt_aa | WT_seq)

    Both probabilities are read off the **same softmax** at the variant
    position, from the unmasked forward pass of the WT sequence. There
    is no MUT-sequence forward pass — the model evaluates both amino
    acid identities against the same WT context. This matches Brandes'
    "one forward pass per variant" recipe.

    Sign convention (matches Brandes):
      - LLR < 0  → deleterious   (MUT less likely than WT under WT-context)
      - LLR ~ 0  → neutral
      - LLR > 0  → tolerated / WT-displacing

    For AUROC with pathogenic = positive class, pass `-llr` to
    sklearn.metrics.roc_auc_score (higher predictor = positive class).

    Indexing: BOS is at token index 0, so sequence position p (1-indexed)
    maps directly to token index p.

    History: prior implementation took a `mut_logits` argument and computed
    a two-pass formula log P_wt(wt|wt_seq) - log P_mut(mut|mut_seq) with
    inverted sign. Both errors were corrected on 2026-05-13 to match
    Brandes 2023 exactly. AUROC was 0.929 with the broken formula and is
    0.930 with this one — the two quantities are highly correlated, but
    the method now matches the paper.
    """
    pos = mutation.position

    def log_softmax(x: np.ndarray) -> np.ndarray:
        x_max = np.max(x)
        shifted = x - x_max
        return shifted - np.log(np.sum(np.exp(shifted)))

    log_probs = log_softmax(wt_logits[pos])
    return float(log_probs[mut_token_id] - log_probs[wt_token_id])


def compute_lid(
    query: np.ndarray, reference: np.ndarray, k: int = 20,
) -> float:
    """Local Intrinsic Dimensionality (Amsaleg 2015 MLE estimator).

        LID = -(k - 1) / sum(log(r_i / r_k))  for i in [1, k-1]

    where r_i is the distance to the i-th nearest neighbor and r_k is the
    distance to the k-th. Lower LID = more constrained local geometry
    (often pathogenic in VEP).
    """
    n_ref = len(reference)
    k = min(k, n_ref - 1)
    if k < 2:
        return float("nan")

    dists = np.linalg.norm(reference - query[None, :], axis=1)
    dists.sort()
    dists = dists[dists > 1e-10]
    if len(dists) < k:
        k = len(dists)
    if k < 2:
        return float("nan")

    nearest = dists[:k]
    r_k = nearest[-1]
    if r_k < 1e-10:
        return float("nan")
    log_ratios = np.log(nearest[:-1] / r_k)
    return float(-(k - 1) / np.sum(log_ratios))


def _percentile_rank(value: float, ref: dict) -> float:
    """Percentile rank via piecewise-linear interpolation over reference quantiles."""
    pairs = []
    for k, v in ref.items():
        if isinstance(k, str) and k.startswith("p") and k[1:].isdigit():
            pairs.append((float(k[1:]), float(v)))
    if len(pairs) < 2:
        return float("nan")
    pairs.sort(key=lambda pv: pv[1])
    quantiles = [p for p, _ in pairs]
    values = [v for _, v in pairs]
    if value <= values[0]:
        return quantiles[0]
    if value >= values[-1]:
        return quantiles[-1]
    for i in range(len(values) - 1):
        if values[i] <= value <= values[i + 1]:
            span = values[i + 1] - values[i]
            if span < 1e-10:
                return quantiles[i]
            frac = (value - values[i]) / span
            return quantiles[i] + frac * (quantiles[i + 1] - quantiles[i])
    return float("nan")


def score_variant_report(
    mutation: MutationSpec,
    metrics: dict,
    reference_distributions: dict,
) -> str:
    """Human-readable report for a BYOD variant.

    Contextualizes each metric against the workshop dataset's reference
    distribution (mean, SD, per-class medians) and flags whether the
    variant looks more pathogenic-like or benign-like on each score.
    """
    lines = [
        f"Variant Effect Report: {mutation}",
        "=" * 60,
        "",
    ]

    metric_map = {
        "delta_norm": ("delta_norm_protein", "Delta norm (L2)", True),
        "cosine_dist": ("cosine_dist_protein", "Cosine distance", True),
        "llr": ("llr_protein", "LLR (log-likelihood ratio)", True),
        "lid": ("lid_protein", "LID (local intrinsic dim)", False),
    }

    for key, (ref_key, label, higher_is_pathogenic) in metric_map.items():
        if key not in metrics:
            continue
        value = metrics[key]
        lines.append(f"• {label}: {value:.3f}")
        if ref_key not in reference_distributions:
            lines.append("    (no reference distribution available)")
            continue
        d = reference_distributions[ref_key]
        pct = _percentile_rank(value, d)
        lines.append(f"    Reference mean: {d['mean']:.3f} ± {d['std']:.3f}")
        lines.append(f"    Percentile rank: {pct:.1f}%")
        if higher_is_pathogenic:
            verdict = (
                "pathogenic-like" if pct >= 75
                else ("benign-like" if pct <= 25 else "intermediate")
            )
        else:
            verdict = (
                "pathogenic-like" if pct <= 25
                else ("benign-like" if pct >= 75 else "intermediate")
            )
        lines.append(f"    Directional verdict: {verdict} (percentile-based)")

        path_med = d.get("pathogenic_median")
        ben_med = d.get("benign_median")
        if path_med is not None and ben_med is not None:
            path_dist = abs(value - path_med)
            ben_dist = abs(value - ben_med)
            closer = "pathogenic" if path_dist < ben_dist else "benign"
            lines.append(
                f"    Class medians: pathogenic={path_med:.3f}, "
                f"benign={ben_med:.3f} → value is closer to **{closer}**"
            )
        lines.append("")

    lines.append("-" * 60)
    lines.append("Note: this is a zero-shot score. Accuracy depends on the model")
    lines.append("and reference dataset. Always cross-check with specialized tools")
    lines.append("like AlphaMissense for clinically-relevant variants.")
    return "\n".join(lines)


__all__ = [
    "AA_ALPHABET",
    "AA_FULL_NAMES",
    "MutationSpec",
    "apply_mutation",
    "compute_cosine_distance",
    "compute_delta_norm",
    "compute_lid",
    "compute_llr",
    "encode_variant",
    "parse_mutation",
    "score_variant_report",
    "truncate_around_mutation",
    "validate_mutation",
    "validate_sequence",
]
