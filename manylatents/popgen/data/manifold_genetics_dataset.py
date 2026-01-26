"""
ManifoldGeneticsDataset - Dataset-agnostic loader for manifold-genetics outputs.

This module provides a PyTorch Dataset that consumes standardized outputs from
the manifold-genetics package, replacing the legacy PLINK-based dataset classes
that contained biobank-specific logic.

Expected manifold-genetics output structure:
    output_dir/
    ├── pca/
    │   ├── fit_pca_*.csv       # PCA coordinates for training samples
    │   └── transform_pca_*.csv # PCA coordinates for test samples
    ├── admixture/
    │   ├── fit.K{k}.csv        # Admixture proportions for training (e.g., fit.K5.csv)
    │   └── transform.K{k}.csv  # Admixture proportions for test
    ├── embeddings/
    │   └── *.csv               # Optional custom embeddings
    ├── labels.csv              # Sample labels with sample_id column
    └── colormap.json           # Label-to-color mapping for visualization

All CSVs are aligned by 'sample_id' column.
"""

import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ManifoldGeneticsDataset(Dataset):
    """
    PyTorch Dataset for manifold-genetics standardized outputs.

    This dataset is completely biobank-agnostic and replaces the legacy
    PLINK-based dataset classes (HGDPDataset, AOUDataset, UKBBDataset, MHIDataset).

    All preprocessing, filtering, and subsetting is handled upstream by manifold-genetics.
    This class only handles:
    - Loading CSVs
    - Joining by sample_id
    - Exposing tensors for PyTorch DataLoaders

    Label Encoding (IMPORTANT):
        Labels are stored as strings in the CSV but returned as integers from get_labels()
        for compatibility with torch.tensor() in manylatents core. The mapping is preserved:

        - get_labels() -> integer-encoded labels (for tensor operations)
        - get_label_names() -> original string labels (for visualization/colormap)
        - get_label_classes() -> unique labels in sorted order (index = integer encoding)
        - _label_encoders -> dict mapping {label_col: {string: int}}
        - _label_classes -> dict mapping {label_col: np.array of strings}

        This is a workaround for a design issue in manylatents core where experiment.py
        expects integer labels for torch.tensor() but PlotEmbeddings expects string labels
        for colormap matching. When manylatents core adds proper label_names support,
        this encoding can be revisited.

    Args:
        pca_path: Path to PCA coordinates CSV (contains sample_id column)
        admixture_paths: Dict mapping K values to admixture CSV paths (e.g., {5: "fit.K5.csv"})
        labels_path: Path to labels.csv (must contain sample_id and label columns)
        colormap_path: Path to colormap.json for visualization
        embedding_path: Optional path to custom embedding CSV
        label_column: Column name in labels.csv to use as labels (default: "Population")

    CSV Format Requirements:
        - All CSVs must have a 'sample_id' column for joining
        - PCA CSV: sample_id, dim_1, dim_2, ..., dim_n
        - Admixture CSV: sample_id, component_1, component_2, ..., component_K
        - Labels CSV: sample_id, <label_column>, [other label columns], [latitude, longitude]
        - Embeddings CSV: sample_id, dim_1, dim_2, ..., dim_n (or custom column names)

    Colormap JSON: Nested dict by label type, e.g.:
          {
            "Population": {"Yoruba": "#FF0000", "Han": "#00FF00"},
            "Genetic_region": {"Africa": "#FF6B6B", "EastAsia": "#4ECDC4"}
          }
    """
    
    def __init__(
        self,
        pca_path: Optional[str] = None,
        admixture_paths: Optional[Dict[int, str]] = None,
        labels_path: Optional[str] = None,
        colormap_path: Optional[str] = None,
        embedding_path: Optional[str] = None,
        label_column: str = "Population",
        geographic_labels_path: Optional[str] = None,
    ):
        super().__init__()

        self.pca_path = pca_path
        self.admixture_paths = admixture_paths or {}
        self.labels_path = labels_path
        self.colormap_path = colormap_path
        self.embedding_path = embedding_path
        self.label_column = label_column
        self.geographic_labels_path = geographic_labels_path

        # Main data (PCA/embeddings) - used as input to algorithms
        self.data_df = None
        self.data_array = None

        # Admixture ratios - stored separately for metrics (not part of input data)
        # Format: {K: DataFrame} where DataFrame has sample_id + component columns
        self.admixture_ratios: Dict[int, pd.DataFrame] = {}

        # Labels and metadata
        self.labels_df = None
        self.colormap = None
        self.sample_ids = None

        # Geographic labels - samples with meaningful geography (for GeographicPreservation metric)
        self._geographic_sample_ids: Optional[set] = None

        # Label encoding: maps string labels to integers for torch.tensor() compatibility
        # See get_labels() docstring for details on why this is needed
        self._label_encoders: Dict[str, Dict[str, int]] = {}  # {col: {label: int}}
        self._label_classes: Dict[str, np.ndarray] = {}  # {col: array of unique labels}

        self._load_data()

    def _load_data(self):
        """
        Load and merge data sources by sample_id.

        Data is separated into:
        - self.data_array: Input features (PCA/embeddings) for algorithms
        - self.admixture_ratios: Admixture proportions for metrics (not input data)
        - self.labels_df: Sample labels for visualization/evaluation
        """
        # Track sample IDs for alignment across all data sources
        master_sample_ids = None

        # --- Load PCA (primary input data) ---
        pca_df = None
        if self.pca_path:
            logger.info(f"Loading PCA from {self.pca_path}")
            pca_df = pd.read_csv(self.pca_path)
            if 'sample_id' not in pca_df.columns:
                raise ValueError(f"PCA CSV must contain 'sample_id' column. Found: {pca_df.columns.tolist()}")
            master_sample_ids = set(pca_df['sample_id'].values)
            dim_cols = [c for c in pca_df.columns if c != 'sample_id']
            logger.info(f"Loaded {len(pca_df)} samples with {len(dim_cols)} PCA dimensions")

        # --- Load custom embedding (additional input data) ---
        embed_df = None
        if self.embedding_path:
            logger.info(f"Loading embedding from {self.embedding_path}")
            embed_df = pd.read_csv(self.embedding_path)
            if 'sample_id' not in embed_df.columns:
                raise ValueError(f"Embedding CSV must contain 'sample_id' column. Found: {embed_df.columns.tolist()}")
            if master_sample_ids is None:
                master_sample_ids = set(embed_df['sample_id'].values)
            else:
                master_sample_ids &= set(embed_df['sample_id'].values)
            logger.info(f"Loaded {len(embed_df)} samples with {len(embed_df.columns)-1} embedding dimensions")

        # --- Load Admixture (for metrics, NOT input data) ---
        if self.admixture_paths:
            for k, admix_path in self.admixture_paths.items():
                logger.info(f"Loading admixture K={k} from {admix_path}")
                admix_df = pd.read_csv(admix_path)
                if 'sample_id' not in admix_df.columns:
                    raise ValueError(f"Admixture CSV must contain 'sample_id' column. Found: {admix_df.columns.tolist()}")
                self.admixture_ratios[k] = admix_df
                if master_sample_ids is None:
                    master_sample_ids = set(admix_df['sample_id'].values)
                else:
                    master_sample_ids &= set(admix_df['sample_id'].values)
                logger.info(f"Loaded {len(admix_df)} samples with K={k} admixture components")

        # --- Load labels ---
        if self.labels_path:
            logger.info(f"Loading labels from {self.labels_path}")
            self.labels_df = pd.read_csv(self.labels_path)
            if 'sample_id' not in self.labels_df.columns:
                raise ValueError(f"Labels CSV must contain 'sample_id' column. Found: {self.labels_df.columns.tolist()}")
            if self.label_column not in self.labels_df.columns:
                raise ValueError(
                    f"Label column '{self.label_column}' not found in labels CSV. "
                    f"Available columns: {self.labels_df.columns.tolist()}"
                )
            if master_sample_ids is None:
                master_sample_ids = set(self.labels_df['sample_id'].values)
            else:
                master_sample_ids &= set(self.labels_df['sample_id'].values)
            logger.info(f"Loaded {len(self.labels_df)} samples with labels")

        # --- Load geographic labels (for GeographicPreservation metric) ---
        # This CSV contains only samples with meaningful geography (pre-filtered upstream)
        # Also provides latitude/longitude for those samples
        self._geographic_df = None
        if self.geographic_labels_path:
            logger.info(f"Loading geographic labels from {self.geographic_labels_path}")
            self._geographic_df = pd.read_csv(self.geographic_labels_path)
            if 'sample_id' not in self._geographic_df.columns:
                raise ValueError(f"Geographic CSV must contain 'sample_id' column. Found: {self._geographic_df.columns.tolist()}")
            self._geographic_sample_ids = set(self._geographic_df['sample_id'].values)
            logger.info(f"Loaded {len(self._geographic_sample_ids)} samples with geographic coordinates")

        # --- Validate we have input data ---
        if pca_df is None and embed_df is None:
            raise ValueError("No input data provided. Must specify pca_path or embedding_path.")

        if master_sample_ids is None or len(master_sample_ids) == 0:
            raise ValueError("No overlapping samples found across data sources.")

        # --- Build master DataFrame with aligned samples ---
        # Start with PCA or embedding as base
        if pca_df is not None:
            self.data_df = pca_df[pca_df['sample_id'].isin(master_sample_ids)].copy()
        else:
            self.data_df = embed_df[embed_df['sample_id'].isin(master_sample_ids)].copy()

        # Merge embedding if both PCA and embedding provided
        if pca_df is not None and embed_df is not None:
            embed_filtered = embed_df[embed_df['sample_id'].isin(master_sample_ids)]
            self.data_df = self.data_df.merge(embed_filtered, on='sample_id', how='inner')

        # Merge labels
        if self.labels_df is not None:
            labels_filtered = self.labels_df[self.labels_df['sample_id'].isin(master_sample_ids)]
            self.data_df = self.data_df.merge(labels_filtered, on='sample_id', how='inner')

        # Merge geographic coordinates (latitude/longitude) from geographic CSV
        # Use left join so samples without geo data get NaN for lat/lon
        if self._geographic_df is not None:
            geo_cols = ['sample_id']
            if 'latitude' in self._geographic_df.columns:
                geo_cols.append('latitude')
            if 'longitude' in self._geographic_df.columns:
                geo_cols.append('longitude')
            if len(geo_cols) > 1:  # Have lat/lon columns
                geo_filtered = self._geographic_df[geo_cols]
                self.data_df = self.data_df.merge(geo_filtered, on='sample_id', how='left')
                logger.info(f"Merged latitude/longitude for {self.data_df['latitude'].notna().sum()} samples")

        # Sort by sample_id for consistent ordering
        self.data_df = self.data_df.sort_values('sample_id').reset_index(drop=True)
        self.sample_ids = self.data_df['sample_id'].values

        # --- Align admixture_ratios to master sample order ---
        # Keep original column names (sample_id, component_1, component_2, etc.)
        for k in self.admixture_ratios:
            admix_df = self.admixture_ratios[k]
            admix_df = admix_df[admix_df['sample_id'].isin(master_sample_ids)]
            admix_df = admix_df.sort_values('sample_id').reset_index(drop=True)
            self.admixture_ratios[k] = admix_df
            component_cols = [c for c in admix_df.columns if c != 'sample_id']
            logger.info(f"Aligned admixture K={k}: {len(admix_df)} samples, {len(component_cols)} components")

        # --- Load colormap ---
        if self.colormap_path:
            logger.info(f"Loading colormap from {self.colormap_path}")
            with open(self.colormap_path, 'r') as f:
                self.colormap = json.load(f)
            if isinstance(self.colormap, dict):
                total_colors = sum(len(v) if isinstance(v, dict) else 0 for v in self.colormap.values())
                logger.info(f"Loaded colormap with {len(self.colormap)} label types, {total_colors} total colors")

        # --- Extract input data array (PCA/embeddings only, NOT admixture) ---
        # Identify which columns are input features vs metadata
        metadata_cols = {'sample_id', 'latitude', 'longitude'}  # Always exclude these
        input_columns = []
        for col in self.data_df.columns:
            if col in metadata_cols:
                continue
            # Skip label/metadata columns from labels CSV
            if self.labels_df is not None and col in self.labels_df.columns:
                continue
            input_columns.append(col)

        self.data_array = self.data_df[input_columns].values.astype(np.float32)
        logger.info(f"Final dataset: {len(self)} samples × {self.data_array.shape[1]} input features")
        logger.info(f"Admixture stored separately for {len(self.admixture_ratios)} K values (for metrics)")
    
    def __len__(self) -> int:
        return len(self.data_df)

    @property
    def data(self) -> np.ndarray:
        """Return data array (required by manylatents core for evaluate_embeddings)."""
        return self.data_array
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            dict: {
                'data': np.ndarray of features (PCA/admixture/embeddings),
                'metadata': dict of sample metadata including labels
            }
        """
        sample_data = self.data_array[idx]
        
        # Get all metadata for this sample
        metadata = self.data_df.iloc[idx].to_dict()
        
        return {
            'data': sample_data,
            'metadata': metadata
        }
    
    def _build_label_encoder(self, label_col: str) -> None:
        """
        Build label encoder for a specific column (lazy initialization).

        Creates a mapping from string labels to integers for torch.tensor() compatibility.
        The mapping is stored in self._label_encoders and self._label_classes.
        """
        if label_col in self._label_encoders:
            return  # Already built

        unique_labels = sorted(self.data_df[label_col].unique())
        self._label_classes[label_col] = np.array(unique_labels)
        self._label_encoders[label_col] = {label: idx for idx, label in enumerate(unique_labels)}
        logger.info(f"Built label encoder for '{label_col}': {len(unique_labels)} unique labels")

    def get_labels(self, label_col: Optional[str] = None) -> np.ndarray:
        """
        Get integer-encoded labels for all samples.

        Labels are encoded as integers for compatibility with torch.tensor() in
        manylatents core (experiment.py:518). String labels would cause:
            TypeError: can't convert np.ndarray of type numpy.object_

        Args:
            label_col: Label column to extract (default: self.label_column)

        Returns:
            np.ndarray: Array of integer-encoded labels (dtype: int64)

        Note:
            To get original string labels for visualization/colormap matching,
            use get_label_names() instead. The mapping between integers and
            strings is stored in self._label_encoders and self._label_classes.

        TODO (manylatents core issue):
            The PlotEmbeddings callback expects string labels for colormap matching,
            but experiment.py expects integer labels for tensor conversion. This is
            a design issue in manylatents core that should be addressed by adding
            proper label_names support. See GitHub issue for details.
        """
        if self.labels_df is None:
            raise ValueError("No labels loaded. Provide labels_path to constructor.")

        label_col = label_col or self.label_column
        if label_col not in self.data_df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. "
                f"Available: {[c for c in self.data_df.columns if c in self.labels_df.columns]}"
            )

        # Build encoder lazily
        self._build_label_encoder(label_col)

        # Encode string labels to integers
        encoder = self._label_encoders[label_col]
        string_labels = self.data_df[label_col].values
        return np.array([encoder[label] for label in string_labels], dtype=np.int64)

    def get_label_names(self, label_col: Optional[str] = None) -> np.ndarray:
        """
        Get original string labels for all samples.

        Use this method when you need string labels for visualization or
        colormap matching. For tensor-compatible integer labels, use get_labels().

        Args:
            label_col: Label column to extract (default: self.label_column)

        Returns:
            np.ndarray: Array of string labels
        """
        if self.labels_df is None:
            raise ValueError("No labels loaded. Provide labels_path to constructor.")

        label_col = label_col or self.label_column
        if label_col not in self.data_df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. "
                f"Available: {[c for c in self.data_df.columns if c in self.labels_df.columns]}"
            )

        return self.data_df[label_col].values

    def get_label_classes(self, label_col: Optional[str] = None) -> np.ndarray:
        """
        Get unique label classes in sorted order.

        The index of each class corresponds to its integer encoding in get_labels().
        i.e., get_label_classes()[i] is the string label for integer i.

        Args:
            label_col: Label column (default: self.label_column)

        Returns:
            np.ndarray: Array of unique string labels in sorted order
        """
        if self.labels_df is None:
            raise ValueError("No labels loaded. Provide labels_path to constructor.")

        label_col = label_col or self.label_column
        self._build_label_encoder(label_col)
        return self._label_classes[label_col]
    
    def get_sample_ids(self) -> np.ndarray:
        """Get sample IDs for all samples."""
        return self.sample_ids
    
    def get_colormap(self) -> Optional[Dict[str, str]]:
        """
        Get colormap for visualization.
        
        If colormap is nested by label type (e.g., {"Population": {...}, "Genetic_region": {...}}),
        returns the colormap for the current label_column.
        Otherwise returns the full colormap.
        
        Returns:
            Dict mapping label values to colors, or None if no colormap loaded
        """
        if self.colormap is None:
            return None
        
        # If colormap is nested by label type, extract the relevant one
        if isinstance(self.colormap, dict) and self.label_column in self.colormap:
            return self.colormap[self.label_column]
        
        # Otherwise return full colormap (backward compatibility)
        return self.colormap
    
    @property
    def latitude(self) -> Optional[pd.Series]:
        """Get latitude coordinates if available (from labels or geographic CSV)."""
        if 'latitude' in self.data_df.columns:
            return self.data_df['latitude']
        return None

    @property
    def longitude(self) -> Optional[pd.Series]:
        """Get longitude coordinates if available (from labels or geographic CSV)."""
        if 'longitude' in self.data_df.columns:
            return self.data_df['longitude']
        return None

    @property
    def geographic_preservation_indices(self) -> Optional[np.ndarray]:
        """
        Get boolean indices for samples with meaningful geography.

        Returns indices of samples in the main dataset that are also present in the
        geographic labels CSV. This is used by the GeographicPreservation metric to
        exclude samples where geography doesn't correlate with genetics (e.g., recent
        migrants, admixed populations).

        The filtering is done upstream by manifold-genetics - this property just
        identifies which samples in our dataset have geographic data.

        Returns:
            np.ndarray: Boolean array of shape (n_samples,) where True indicates
                        the sample has meaningful geographic coordinates.
            None: If no geographic_labels_path was provided.
        """
        if self._geographic_sample_ids is None:
            return None
        return np.array([sid in self._geographic_sample_ids for sid in self.sample_ids])

    @property
    def population_label(self) -> pd.Series:
        """Compatibility shim for popgen metrics expecting population_label."""
        if self.labels_df is None:
            raise ValueError("No labels loaded. Provide labels_path to constructor.")
        if self.label_column not in self.data_df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found. "
                f"Available: {[c for c in self.data_df.columns if c in self.labels_df.columns]}"
            )
        return self.data_df[self.label_column]
