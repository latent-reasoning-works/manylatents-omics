import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

class PrecomputedMixin:
    """
    Mixin to handle logic for loading precomputed embeddings alongside raw data.
    """
    def load_precomputed(self, precomputed_path: str, mmap_mode: str = None):
        if precomputed_path:
            abs_path = os.path.abspath(precomputed_path)
            logger.info(f"Resolved precomputed path: {abs_path}")
            if os.path.exists(abs_path):
                logger.info(f"Loading precomputed embeddings from {abs_path}")
                data = None
                if abs_path.endswith(".npy"):
                    data = np.load(abs_path, mmap_mode=mmap_mode)
                elif abs_path.endswith(".csv"):
                    # Use genfromtxt to skip the header and select numeric columns.
                    # Adjust `usecols` as needed; here we assume columns 0 and 1 are numeric.
                    data = np.genfromtxt(abs_path, delimiter=",", skip_header=1)
                else:
                    raise ValueError(f"Unsupported file format: {abs_path}")

                # Check for NaN values and warn
                if data is not None:
                    nan_mask = np.any(np.isnan(data), axis=1)
                    nan_count = nan_mask.sum()
                    if nan_count > 0:
                        logger.warning(
                            f"Precomputed data contains {nan_count} samples with NaN values "
                            f"(out of {data.shape[0]} total samples). "
                            f"These may cause errors during training if not filtered out via metadata filters."
                        )

                return data
        logger.info("No precomputed embeddings found or path does not exist.")
        return None

    def get_data(self, idx: int):
        """
        Returns both raw data and precomputed embeddings (if available) for a given index.
        
        Override this method if you need a different return structure.
        """
        # Assume self.original_data is already loaded by the dataset.
        sample_raw = self.original_data[idx]
        sample_precomputed = None
        if hasattr(self, "precomputed_embeddings") and self.precomputed_embeddings is not None:
            sample_precomputed = self.precomputed_embeddings[idx]
        return sample_raw, sample_precomputed