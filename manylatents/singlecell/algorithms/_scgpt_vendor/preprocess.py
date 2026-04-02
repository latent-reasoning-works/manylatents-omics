# Vendored from scGPT (MIT license) — only binning utilities needed for inference.

import logging
from typing import Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """Digitize the data into bins, spreading uniformly when bins have same values."""
    assert x.ndim == 1 and bins.ndim == 1
    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits
    right_digits = np.digitize(x, bins, right=True)
    rands = np.random.rand(len(x))
    digits = rands * (right_digits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Bin the row into n_bins."""
    dtype = row.dtype
    return_np = not isinstance(row, torch.Tensor)
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row

    if row.max() == 0:
        logger.warning("Input row is all zeros.")
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)
