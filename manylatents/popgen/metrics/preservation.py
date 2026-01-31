import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.utils.metrics import (
    compute_average_smoothness,
    compute_geodesic_distances,
    compute_knn_laplacian,
    haversine_vectorized,
)
# Import core preservation functions from manylatents
from manylatents.metrics.preservation import (
    preservation_metric,
    _scale_embedding_dimensions,
)

logger = logging.getLogger(__name__)


##############################################################################
# 2) Geographic Metric
##############################################################################

def compute_geographic_metric(
    ancestry_coords,
    latitude,
    longitude,
    use_medians=False,
    only_far=False,
    subset_to_test_on=None
):
    """
    Compare ground-truth haversine distances (lat/lon) vs. embedding distances (ancestry_coords).
    Return Spearman correlation.
    If use_medians=True, group by lat/lon and take median embedding coords.
    If only_far=True, ignore pairs under the 10th percentile of geo distance.
    """
    if subset_to_test_on is not None:
        ancestry_coords = ancestry_coords[subset_to_test_on]
        latitude = latitude[subset_to_test_on]
        longitude = longitude[subset_to_test_on]

    # Combine lat/lon in a DataFrame
    ground_truth_coords = pd.concat([latitude, longitude], axis=1)
    ground_truth_coords_rad = np.radians(ground_truth_coords)

    combined = pd.concat([
        ground_truth_coords_rad,
        pd.DataFrame(ancestry_coords, index=ground_truth_coords.index)
    ], axis=1)

    if use_medians:
        # group by lat/lon, median the embedding columns
        combined = combined.groupby(["latitude", "longitude"]).median().reset_index()

    # Extract arrays
    ground_truth_arr = combined[["latitude", "longitude"]].values
    ancestry_coords_arr = combined[[0, 1]].values  # two columns for embedding

    # Haversine distances => ground truth
    gt_dists_square = haversine_vectorized(ground_truth_arr)
    gt_dists = squareform(gt_dists_square)

    # Embedding (ancestry) distances
    ac_dists = pdist(ancestry_coords_arr)

    return preservation_metric(gt_dists, ac_dists, only_far=only_far)

##############################################################################
# 3) Admixture-based Metrics
##############################################################################

def compute_continental_admixture_metric_dists(
    ancestry_coords,
    admixture_ratios,
    population_label,
    use_medians=False,
    only_far=False,
    subset_to_test_on=None
):
    """
    Build geodesic distance graph for admixture_ratios, compare vs. ancestry_coords distances.
    If disconnected at k=5, increase k in steps of 5 up to 100. Return Spearman correlation.

    Args:
        ancestry_coords: numpy array of embedding coordinates
        admixture_ratios: DataFrame with sample_id + component columns (e.g., component_1, component_2)
        population_label: Series with population labels
        use_medians: If True, aggregate by population median
        only_far: If True, only consider distant pairs
        subset_to_test_on: Boolean array for subsetting
    """
    if subset_to_test_on is not None:
        ancestry_coords = ancestry_coords[subset_to_test_on]
        admixture_ratios = admixture_ratios.iloc[subset_to_test_on].reset_index(drop=True)
        population_label = population_label.iloc[subset_to_test_on]

    df1 = pd.DataFrame(ancestry_coords,
                       index=population_label.index)
    df1 = df1.rename(columns={i: f'emb{i}' for i in range(ancestry_coords.shape[1])})

    # Get component columns (exclude sample_id)
    component_cols = [c for c in admixture_ratios.columns if c != 'sample_id']
    admixture_values = admixture_ratios[component_cols].dropna()
    # Rename to ar0, ar1, etc. for consistent column matching later
    admixture_values.columns = [f'ar{i}' for i in range(len(component_cols))]

    df = pd.concat([df1.reset_index(drop=True), admixture_values.reset_index(drop=True)], axis=1)
    df = pd.concat([df, population_label.reset_index(drop=True)], axis=1)

    if use_medians:
        df = df.groupby(population_label.name).median().reset_index()

    ancestry_coords2 = df[df.columns[df.columns.map(str).str.startswith('emb')]].values
    admixture_ratios2 = df[df.columns[df.columns.map(str).str.startswith('ar')]].values

    ancestry_dists = pdist(ancestry_coords2)

    k = 5
    while k <= 100:
        admixture_dists = compute_geodesic_distances(admixture_ratios2, k=k, metric='euclidean')
        if admixture_dists is None:
            k += 5
        else:
            return preservation_metric(admixture_dists, ancestry_dists, only_far=only_far)

    return None  # if never got a connected graph


def compute_continental_admixture_metric_laplacian(
    ancestry_coords,
    admixture_ratios,
    subset_to_test_on=None
):
    """
    Evaluate smoothness x^T L x over adjacency built from ancestry_coords.
    Return average across admixture components.

    Args:
        ancestry_coords: numpy array of embedding coordinates
        admixture_ratios: DataFrame with sample_id + component columns, or numpy array
        subset_to_test_on: Boolean array for subsetting
    """
    if subset_to_test_on is not None:
        ancestry_coords = ancestry_coords[subset_to_test_on]
        if isinstance(admixture_ratios, pd.DataFrame):
            admixture_ratios = admixture_ratios.iloc[subset_to_test_on]
        else:
            admixture_ratios = admixture_ratios[subset_to_test_on]

    # Convert DataFrame to numpy array, excluding sample_id column
    if isinstance(admixture_ratios, pd.DataFrame):
        component_cols = [c for c in admixture_ratios.columns if c != 'sample_id']
        admixture_values = admixture_ratios[component_cols].values
    else:
        admixture_values = admixture_ratios

    laplacian = compute_knn_laplacian(ancestry_coords, k=5, normalized=True)
    return compute_average_smoothness(laplacian, admixture_values)

##############################################################################
# 4) Ground Truth based metrics (moved to core manylatents)
##############################################################################
# compute_ground_truth_preservation() and GroundTruthPreservation() are now in
# manylatents.metrics.preservation for generic use with any ground truth distances


##############################################################################
# 5) Example Aggregators
##############################################################################

def compute_k_admixture_metric_dists(
    ancestry_coords,
    admixtures_k,
    admixture_ratios_list,
    population_label,
    subset_to_test_on=None
):
    """
    For multiple 'k' admixture ratio sets, run compute_continental_admixture_metric_dists
    and return a dict of results keyed by k.
    """
    results = {}
    for k_val, admixture_ratios_k in zip(admixtures_k, admixture_ratios_list):
        key = f"admixture_preservation_k={k_val}"

        results[key] = compute_continental_admixture_metric_dists(
            ancestry_coords,
            admixture_ratios_k,
            population_label,
            use_medians=False,
            only_far=False,
            subset_to_test_on=subset_to_test_on
        )
    return results


def compute_quality_metrics(
    ancestry_coords,
    latitude,
    longitude,
    admixtures_k,
    admixture_ratios_list,
    population_label,
    subset_to_test_on=None
):
    """
    High-level aggregator for multiple metrics, returns a dictionary.
    """
    metrics_dict = {
        "geographic_preservation": compute_geographic_metric(
            ancestry_coords, latitude, longitude, 
            use_medians=False, only_far=False,
            subset_to_test_on=subset_to_test_on
        ),
        "geographic_preservation_medians": compute_geographic_metric(
            ancestry_coords, latitude, longitude,
            use_medians=True, only_far=False,
            subset_to_test_on=subset_to_test_on
        ),
        "geographic_preservation_far": compute_geographic_metric(
            ancestry_coords, latitude, longitude,
            use_medians=False, only_far=True,
            subset_to_test_on=subset_to_test_on
        ),
    }

    # Admixture aggregator
    admixture_dict = compute_k_admixture_metric_dists(
        ancestry_coords,
        admixtures_k,
        admixture_ratios_list,
        population_label,
        subset_to_test_on=subset_to_test_on
    )
    metrics_dict.update(admixture_dict)

    return metrics_dict


##############################################################################
# 6) Single-Value Wrappers (conform to Metric(Protocol))
##############################################################################
# _scale_embedding_dimensions() is now imported from manylatents.metrics.preservation

def GeographicPreservation(embeddings: np.ndarray,
                           dataset,
                           module: Optional[LatentModule] = None,
                           scale_embeddings: bool = True,
                           **kwargs) -> float:
    """
    Minimal wrapper that passes extra keyword arguments to compute_geographic_metric.
    """
    if scale_embeddings:
        embeddings = _scale_embedding_dimensions(embeddings)

    return compute_geographic_metric(
        ancestry_coords=embeddings,
        latitude=dataset.latitude,
        longitude=dataset.longitude,
        subset_to_test_on=dataset.geographic_preservation_indices,
        **kwargs
    )

def AdmixturePreservation(embeddings: np.ndarray,
                          dataset,
                          module: Optional[LatentModule] = None,
                          scale_embeddings: bool = True,
                          admixture_k: int = 5,
                          **kwargs) -> float:
    """
    Single-value wrapper returning Spearman correlation for admixture preservation.

    Args:
        embeddings: Embedding coordinates
        dataset: Dataset with admixture_ratios attribute
        module: Optional LatentModule (unused)
        scale_embeddings: Whether to scale embedding dimensions
        admixture_k: K value for admixture proportions (default: 5)
    """
    if scale_embeddings:
        embeddings = _scale_embedding_dimensions(embeddings)

    # Handle both int and string keys for backward compatibility
    k_key = admixture_k
    if k_key not in dataset.admixture_ratios:
        k_key = str(admixture_k)
    if k_key not in dataset.admixture_ratios:
        available = list(dataset.admixture_ratios.keys())
        raise ValueError(f"Admixture K={admixture_k} not found. Available: {available}")

    return compute_continental_admixture_metric_dists(
        ancestry_coords=embeddings,
        admixture_ratios=dataset.admixture_ratios[k_key],
        population_label=dataset.population_label,
        **kwargs
    )

def AdmixturePreservationK(embeddings: np.ndarray,
                           dataset,
                           module: Optional[LatentModule] = None,
                           scale_embeddings: bool = True,
                           max_samples: Optional[int] = None,
                           random_seed: int = 42,
                           **kwargs) -> np.ndarray:
    """
    A vector-value wrapper returning admixture preservation scores for all Ks.

    Args:
        embeddings: Embedding coordinates
        dataset: Dataset with admixture_ratios attribute
        module: Optional LatentModule (unused)
        scale_embeddings: Whether to scale embedding dimensions
        max_samples: If specified, randomly subsample to this many samples (for large datasets)
        random_seed: Random seed for subsampling reproducibility

    Returns:
        Array of preservation scores, one per K value
    """
    n_samples = embeddings.shape[0]

    # Subsample if requested
    if max_samples is not None and n_samples > max_samples:
        rng = np.random.default_rng(random_seed)
        subset_indices = np.sort(rng.choice(n_samples, size=max_samples, replace=False))
        logger.info(f"Subsampling {max_samples} of {n_samples} samples for admixture preservation")
    else:
        subset_indices = None

    if subset_indices is not None:
        embeddings = embeddings[subset_indices]

    if scale_embeddings:
        embeddings = _scale_embedding_dimensions(embeddings)

    return_vector = np.zeros(len(dataset.admixture_ratios))
    for i, key in enumerate(dataset.admixture_ratios.keys()):
        # Subsample admixture and labels if needed
        if subset_indices is not None:
            admixture_df = dataset.admixture_ratios[key].iloc[subset_indices].reset_index(drop=True)
            population_label = dataset.population_label.iloc[subset_indices].reset_index(drop=True)
        else:
            admixture_df = dataset.admixture_ratios[key]
            population_label = dataset.population_label

        result = compute_continental_admixture_metric_dists(
            ancestry_coords=embeddings,
            admixture_ratios=admixture_df,
            population_label=population_label,
            **kwargs
        )
        return_vector[i] = result if result is not None else np.nan
    return return_vector

def AdmixtureLaplacian(embeddings: np.ndarray,
                       dataset,
                       module: Optional[LatentModule] = None,
                       scale_embeddings: bool = True,
                       admixture_k: int = 5) -> float:
    """
    Laplacian-based metric -> single float for callback usage.

    Args:
        embeddings: Embedding coordinates
        dataset: Dataset with admixture_ratios attribute
        module: Optional LatentModule (unused)
        scale_embeddings: Whether to scale embedding dimensions
        admixture_k: K value for admixture proportions (default: 5)
    """
    if scale_embeddings:
        embeddings = _scale_embedding_dimensions(embeddings)

    # Handle both int and string keys for backward compatibility
    k_key = admixture_k
    if k_key not in dataset.admixture_ratios:
        k_key = str(admixture_k)
    if k_key not in dataset.admixture_ratios:
        available = list(dataset.admixture_ratios.keys())
        raise ValueError(f"Admixture K={admixture_k} not found. Available: {available}")

    return compute_continental_admixture_metric_laplacian(
        ancestry_coords=embeddings,
        admixture_ratios=dataset.admixture_ratios[k_key]
    )

# GroundTruthPreservation() has been moved to manylatents.metrics.preservation
# for generic use with any dataset having ground truth distances
