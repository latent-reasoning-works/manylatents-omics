"""Scalar features of how a model's hidden state at a site responds to its alleles.

Each function takes hidden states as arrays whose last axis is the hidden
dimension; any leading axes (variants, layers, repeats) are kept. Everything is
computed in float64 and returned as float64, so a caller that stores states in
float16 should compute these first, from the float32 states.

The functions are model-agnostic: AlphaGenome and Evo2 site states both use them.
"""
from __future__ import annotations

import numpy as np


def _as_float(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def delta_norm(h_ref, h_alt) -> np.ndarray:
    """Euclidean norm of ``h_alt - h_ref`` along the last axis."""
    return np.linalg.norm(_as_float(h_alt) - _as_float(h_ref), axis=-1)


def cosine(a, b) -> np.ndarray:
    """Cosine similarity of ``a`` and ``b`` along the last axis; NaN where a norm is 0."""
    a, b = _as_float(a), _as_float(b)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(norms > 0, dot / np.where(norms > 0, norms, 1.0), np.nan)


def participation_ratio(displacements) -> np.ndarray:
    """Participation ratio of the centred rows of ``displacements`` ([..., k, d]).

    With ``Dc`` the rows minus their mean and ``lambda`` the squared singular
    values of ``Dc``, this is ``(sum lambda)^2 / sum lambda^2``: 1 when the rows
    lie on a line, ``k - 1`` at most. NaN when all rows are equal.

    ``lambda`` are the eigenvalues of the Gram matrix ``G = Dc Dc^T``, so the
    ratio is ``trace(G)^2 / ||G||_F^2`` and needs no decomposition.
    """
    d = _as_float(displacements)
    centred = d - d.mean(axis=-2, keepdims=True)
    gram = centred @ np.swapaxes(centred, -1, -2)
    trace = np.trace(gram, axis1=-2, axis2=-1)
    frobenius2 = np.sum(gram * gram, axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(frobenius2 > 0, trace ** 2 / np.where(frobenius2 > 0, frobenius2, 1.0),
                        np.nan)


def pr3(h_ref, h_alt, h_alt2, h_alt3) -> np.ndarray:
    """Participation ratio of the three alternate-allele displacements from ``h_ref``.

    The rows are ``h_alt - h_ref``, ``h_alt2 - h_ref`` and ``h_alt3 - h_ref``.
    Centring removes ``h_ref``, so this measures the spread of the three
    alternate states around their mean. Lies in [1, 2]: three centred points
    span at most a plane. For eigenvalues ``l1 >= l2`` and ``r = l2 / l1`` it
    equals ``(1 + r)^2 / (1 + r^2)``. NaN when the three alternate states coincide.
    """
    ref = _as_float(h_ref)
    rows = np.stack([_as_float(h) - ref for h in (h_alt, h_alt2, h_alt3)], axis=-2)
    return participation_ratio(rows)
