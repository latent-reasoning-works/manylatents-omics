"""scRNA preprocessing — manylatents-omics#60.

What is under test is the CONTRACT, not the arithmetic: scanpy owns the maths and these are
thin delegations. The three properties the consumer depends on are the ones worth pinning —
a filter returns a mask and removes nothing, nothing mutates its input, and the sparse path
stays sparse — because each of them fails SILENTLY when it breaks.
"""
from __future__ import annotations

import numpy as np
import pytest

sc = pytest.importorskip("scanpy")
sp = pytest.importorskip("scipy.sparse")

from manylatents.singlecell import preprocessing as pp  # noqa: E402


def _counts(n_cells=6, n_genes=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)


def _realistic(n_cells=300, n_genes=80, seed=0):
    """Big enough for the two operations with internal model fits.

    `sc.pp.scrublet` runs a PCA at `n_components=30`, and `highly_variable_genes(seurat_v3)`
    fits a loess over the mean-variance trend — both raise on a toy matrix (`n_components=30
    must be between 1 and min(n_samples, n_features)`; `There are other near singularities`).
    Per-gene means vary so the trend the loess fits is real rather than flat noise.
    """
    rng = np.random.default_rng(seed)
    means = rng.uniform(0.5, 20.0, size=n_genes)
    return rng.poisson(means, size=(n_cells, n_genes)).astype(np.float32)


def _overdispersed(n_cells=800, n_genes=500, dispersion=1.5, seed=0):
    """What `sc.pp.scrublet` needs, which is more than the fixture above.

    Scrublet filters genes by variability BEFORE its own PCA, and Poisson counts are not
    over-dispersed enough to survive it: MEASURED, a 300 x 80 Poisson matrix leaves 9 genes and
    the PCA raises `n_components=30 must be between 1 and min(n_samples, n_features)=9`. Negative
    binomial at this size leaves enough, and flags ~9 % of cells — the shape of a real result
    rather than a degenerate one.
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.5, 30.0, size=n_genes)
    p = dispersion / (dispersion + mu)
    return rng.negative_binomial(dispersion, p, size=(n_cells, n_genes)).astype(np.float32)


# ── the mask contract ────────────────────────────────────────────────────────
def test_filter_cells_keeps_cells_above_the_declared_gene_count():
    """Row 1 expresses one gene of three. `min_genes` is the caller's declaration, not this
    module's recommendation — measured on pbmc3k the tutorial's 200 drops 0 of 2700 cells."""
    counts = np.array([[1, 1, 1], [0, 0, 5], [2, 3, 4]], dtype=np.float32)

    keep = pp.filter_cells(counts, min_genes=2)

    assert keep.dtype == bool
    assert keep.tolist() == [True, False, True]


def test_filter_cells_can_threshold_within_groups():
    """Library size differs systematically between collection days, so one global percentile
    drops a whole timepoint rather than the worst cells in each. Cells 0 and 2 are the small
    ones in their own group; a global cut would have taken both cells of group d0."""
    counts = np.array([[1], [10], [1], [10]], dtype=np.float32)
    groups = np.array(["d0", "d0", "d9", "d9"])

    keep = pp.filter_cells(counts, pct_library_size=50, groups=groups)

    assert keep.tolist() == [False, True, False, True]


def test_a_group_label_per_cell_or_it_refuses():
    """The one way a caller silently gets a wrong answer here is a mismatched group vector,
    which numpy would broadcast rather than reject."""
    with pytest.raises(ValueError, match="one group label per cell"):
        pp.filter_cells(_counts(6), pct_library_size=50, groups=np.array(["a", "b"]))


def test_filter_genes_masks_the_gene_axis():
    """Measured on pbmc3k: 19,024 of 32,738 genes are present in fewer than 3 cells. Gene 1 is
    detected in none."""
    counts = np.array([[1, 0, 1], [1, 0, 0], [1, 0, 1]], dtype=np.float32)

    keep = pp.filter_genes(counts, min_cells=2)

    assert keep.tolist() == [True, False, True]
    assert keep.shape[0] == counts.shape[1], "a gene mask is over COLUMNS"


def test_filter_mito_uses_the_names_and_the_declared_prefix():
    """Cell 1 is 50 % mitochondrial and exceeds a 5 % threshold. Measured on pbmc3k: 13 genes
    match `MT-`, median 2.03 %, 57 cells above 5 %."""
    counts = np.array([[99.0, 1.0], [5.0, 5.0]])   # 1 % and 50 %
    genes = ["GAPDH", "MT-CO1"]

    assert pp.filter_mito(counts, genes, max_pct=5).tolist() == [True, False]


def test_filter_mito_refuses_without_names_rather_than_matching_nothing():
    """A prefix match against a nameless matrix selects no genes and reports that it removed no
    cells — indistinguishable from a dataset with no dying cells. That conflation is the reason
    the signature takes names at all."""
    with pytest.raises(ValueError, match="needs gene names"):
        pp.filter_mito(_counts(), None, max_pct=5)


def test_filter_mito_refuses_a_name_per_column_mismatch():
    with pytest.raises(ValueError, match="one name per column"):
        pp.filter_mito(_counts(6, 4), ["a", "b"], max_pct=5)


def test_a_cell_with_no_counts_is_not_called_mitochondrial():
    """0/0. Left in rather than dropped: an empty cell is a filter_cells question, and returning
    NaN here would propagate into a comparison that is False for both directions."""
    counts = np.array([[0.0, 0.0], [99.0, 1.0]])   # no counts, and 1 %

    assert pp.filter_mito(counts, ["GAPDH", "MT-CO1"], max_pct=5).tolist() == [True, True]


# ── nothing mutates its input ────────────────────────────────────────────────
@pytest.mark.parametrize("call", [
    lambda c: pp.normalize(c, target_sum=10.0),
    lambda c: pp.transform(c, method="log1p"),
    lambda c: pp.filter_cells(c, min_genes=1),
    lambda c: pp.filter_genes(c, min_cells=1),
    lambda c: pp.highly_variable_genes(c, n_top_genes=8),
])
def test_no_operation_writes_to_the_matrix_it_was_given(call):
    """`sc.pp.normalize_total` writes IN PLACE, and the matrix a caller passes is typically the
    user's own `.X`. Every scanpy call that could write goes through a copy — this is what
    watches that, across every operation rather than only the obvious one."""
    counts = _realistic()
    before = counts.copy()

    call(counts)

    assert np.array_equal(counts, before)


def test_normalize_scales_each_cell_to_the_declared_library_size():
    counts = np.array([[1.0, 1.0], [2.0, 6.0]])

    out = pp.normalize(counts, target_sum=10.0)

    assert np.allclose(out.sum(axis=1), [10.0, 10.0])


def test_transform_reads_the_working_matrix_not_the_counts():
    """`normalize` then `transform` must compose. A transform that re-read the raw counts would
    discard the normalization in a caller that believes it did both — and nothing downstream
    would show it."""
    counts = np.array([[100.0, 100.0]])

    out = pp.transform(pp.normalize(counts, target_sum=1.0), method="log1p")

    assert np.allclose(out, np.log1p([[0.5, 0.5]]))


def test_an_unknown_transform_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="must be one of"):
        pp.transform(_counts(), method="lg1p")


def test_sqrt_is_available_because_the_EB_convention_uses_it():
    assert np.allclose(pp.transform(np.array([[4.0, 9.0]]), method="sqrt"), [[2.0, 3.0]])


# ── the sparse path ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("fn,kwargs", [
    (pp.filter_cells, {"min_genes": 2}),
    (pp.filter_genes, {"min_cells": 2}),
])
def test_the_filters_answer_identically_on_a_sparse_matrix(fn, kwargs):
    """pbmc3k is an 18.3 MB CSR that densifies to 353.6 MB. A filter that calls `_dense` to
    count nonzeros has made the cheap path expensive for a column sum."""
    dense = _counts(30, 8)
    sparse = sp.csr_matrix(dense)

    assert np.array_equal(fn(dense, **kwargs), fn(sparse, **kwargs))


def test_filter_mito_answers_identically_on_a_sparse_matrix():
    dense = _counts(30, 4)
    genes = ["GAPDH", "MT-CO1", "ACTB", "MT-ND1"]

    assert np.array_equal(pp.filter_mito(dense, genes, max_pct=40),
                          pp.filter_mito(sp.csr_matrix(dense), genes, max_pct=40))


# ── the detection the consumer needs before normalizing ──────────────────────
def test_looks_like_counts_separates_raw_from_transformed():
    """Normalizing an already-normalized matrix is silent and wrong, and no downstream number
    reveals it. Exported so the caller can ask; `normalize` deliberately does not ask for it."""
    assert pp.looks_like_counts(_counts(30, 5)) is True
    assert pp.looks_like_counts(np.log1p(_counts(30, 5))) is False


def test_looks_like_counts_is_false_for_what_it_cannot_inspect():
    """The conservative direction: a False leaves the matrix alone."""
    assert pp.looks_like_counts(object()) is False


# ── doublets: flagged, not removed ───────────────────────────────────────────
def test_detect_doublets_returns_is_doublet_not_keep():
    """The inverse of the filters, deliberately: the consumer's default is to flag without
    removing, so it inverts when it wants a keep-mask. On this synthetic matrix scrublet should
    call almost nothing a doublet — what is pinned is the ORIENTATION and the shape, not the
    call rate, which is scanpy's business."""
    counts = _overdispersed()

    is_doublet = pp.detect_doublets(counts)

    assert is_doublet.dtype == bool
    assert is_doublet.shape == (counts.shape[0],)
    assert is_doublet.sum() < counts.shape[0] / 2, (
        "orientation is is_doublet; a majority would mean it returned a keep-mask")


# ── HVG ──────────────────────────────────────────────────────────────────────
def test_highly_variable_genes_masks_the_gene_axis_and_takes_n_top():
    counts = _realistic()

    mask = pp.highly_variable_genes(counts, n_top_genes=8)

    assert mask.dtype == bool
    assert mask.shape == (counts.shape[1],), "an HVG mask is over COLUMNS"
    assert mask.sum() == 8


def test_hvg_here_is_across_cells_and_not_across_time():
    """A guard against the name collision rather than the arithmetic. The HVG a trajectory
    method wants selects on variability along a TIME axis and takes a `(T, n_genes)` matrix;
    this one selects across cells. They share a name and nothing else — if both ever live in
    this module they need different ones."""
    import inspect

    doc = inspect.getdoc(pp.highly_variable_genes)

    assert "ACROSS CELLS" in doc
    assert "time" in doc.lower(), "the collision must stay named where someone will read it"


def test_normalize_preserves_sparsity_and_does_not_wrap_it():
    """REGRESSION, and the contract the module's own header states.

    `np.asarray()` on a scipy sparse matrix WRAPS rather than converts — it returns a
    0-dimensional object array holding the matrix. Measured on a 20 x 6 CSR before the fix:
    `shape ()`, `dtype object`, so every downstream shape check silently saw `()` and the
    values were unreachable. Densifying instead would be correct and wrong for a different
    reason: pbmc3k goes 18.3 MB -> 353.6 MB.
    """
    dense = _counts(20, 6)
    sparse = sp.csr_matrix(dense)

    out = pp.normalize(sparse, target_sum=10.0)

    assert sp.issparse(out), f"sparse in, sparse out — got {type(out).__name__} {getattr(out, 'shape', None)}"
    assert out.shape == (20, 6)
    assert np.allclose(np.asarray(out.sum(axis=1)).ravel(), 10.0)


def test_normalize_agrees_on_dense_and_sparse_input():
    """The two paths must not diverge numerically — the whole point of preserving sparsity is
    that it changes the representation and nothing else."""
    dense = _counts(20, 6)

    from_dense = pp.normalize(dense, target_sum=10.0)
    from_sparse = pp.normalize(sp.csr_matrix(dense), target_sum=10.0)

    assert np.allclose(np.asarray(from_dense), from_sparse.toarray(), atol=1e-5)
