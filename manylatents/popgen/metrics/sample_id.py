import logging
import numpy as np
from typing import Optional
from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


def SampleId(
    embeddings: np.ndarray,
    dataset,
    module: Optional[LatentModule] = None,
    random_state: int = 42
) -> np.ndarray:
    """
    Fetch sample IDs from dataset for downstream analysis.

    Parameters
    ----------
    embeddings : np.ndarray
        Low-dimensional embeddings (n_samples × n_components)
    dataset : object
        Dataset object, optionally with get_sample_ids() method
    module : LatentModule, optional
        Algorithm module (unused, for protocol compatibility)
    random_state : int, optional
        Random state (unused, for protocol compatibility)

    Returns
    -------
    np.ndarray
        Array of sample IDs
    """
    if hasattr(dataset, 'get_sample_ids'):
        return dataset.get_sample_ids()
    else:
        # Return array of indices as sample IDs when not available
        logger.warning(
            "Sample IDs not found in dataset (no get_sample_ids method). "
            "Using numeric indices instead."
        )
        return np.arange(len(embeddings))
