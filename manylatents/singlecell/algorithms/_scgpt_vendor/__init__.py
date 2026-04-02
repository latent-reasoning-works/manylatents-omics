# Vendored minimal subset of scGPT (bowang-lab/scGPT, MIT license)
# for zero-shot cell embedding extraction only.
#
# Only the forward inference path is included — no training, no GRN,
# no perturbation prediction.  The torchtext dependency is replaced
# with a plain dict-based vocabulary.
#
# Reference:
#   Cui, H. et al. scGPT: toward building a foundation model for
#   single-cell multi-omics using generative AI. Nature Methods (2024).
