"""
Tests for ManifoldGeneticsDataset.

These tests verify the dataset-agnostic loader for manifold-genetics outputs.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manylatents.popgen.data import ManifoldGeneticsDataset


@pytest.fixture
def temp_manifold_dir():
    """Create temporary directory with mock manifold-genetics outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create sample IDs
        sample_ids = [f"sample_{i:03d}" for i in range(100)]
        
        # Create PCA CSV
        pca_data = {
            'sample_id': sample_ids,
            'PC1': np.random.randn(100),
            'PC2': np.random.randn(100),
            'PC3': np.random.randn(100),
        }
        pca_df = pd.DataFrame(pca_data)
        pca_path = tmpdir / "pca.csv"
        pca_df.to_csv(pca_path, index=False)
        
        # Create Admixture CSVs for K=3 and K=5
        admix_k3_data = {
            'sample_id': sample_ids,
            'Ancestry1': np.random.dirichlet([1, 1, 1], 100)[:, 0],
            'Ancestry2': np.random.dirichlet([1, 1, 1], 100)[:, 1],
            'Ancestry3': np.random.dirichlet([1, 1, 1], 100)[:, 2],
        }
        admix_k3_df = pd.DataFrame(admix_k3_data)
        admix_k3_path = tmpdir / "admix_k3.csv"
        admix_k3_df.to_csv(admix_k3_path, index=False)
        
        admix_k5_data = {
            'sample_id': sample_ids,
            **{f'Ancestry{i}': np.random.dirichlet([1]*5, 100)[:, i-1] for i in range(1, 6)}
        }
        admix_k5_df = pd.DataFrame(admix_k5_data)
        admix_k5_path = tmpdir / "admix_k5.csv"
        admix_k5_df.to_csv(admix_k5_path, index=False)
        
        # Create Labels CSV
        labels_data = {
            'sample_id': sample_ids,
            'Population': [f'Pop{i % 5}' for i in range(100)],
            'latitude': np.random.uniform(-90, 90, 100),
            'longitude': np.random.uniform(-180, 180, 100),
        }
        labels_df = pd.DataFrame(labels_data)
        labels_path = tmpdir / "labels.csv"
        labels_df.to_csv(labels_path, index=False)
        
        # Create colormap JSON
        colormap = {
            'Pop0': '#FF0000',
            'Pop1': '#00FF00',
            'Pop2': '#0000FF',
            'Pop3': '#FFFF00',
            'Pop4': '#FF00FF',
        }
        colormap_path = tmpdir / "colormap.json"
        with open(colormap_path, 'w') as f:
            json.dump(colormap, f)
        
        yield {
            'dir': tmpdir,
            'pca_path': str(pca_path),
            'admix_k3_path': str(admix_k3_path),
            'admix_k5_path': str(admix_k5_path),
            'labels_path': str(labels_path),
            'colormap_path': str(colormap_path),
            'sample_ids': sample_ids,
        }


def test_dataset_init_pca_only(temp_manifold_dir):
    """Test loading dataset with PCA only."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
    )
    
    assert len(dataset) == 100
    assert dataset.data_array.shape == (100, 3)  # 3 PCs
    
    # Test getting a sample
    sample = dataset[0]
    assert 'data' in sample
    assert 'metadata' in sample
    assert sample['data'].shape == (3,)
    assert 'sample_id' in sample['metadata']


def test_dataset_init_pca_and_admixture(temp_manifold_dir):
    """Test loading dataset with PCA and admixture."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        admixture_paths={
            3: temp_manifold_dir['admix_k3_path'],
            5: temp_manifold_dir['admix_k5_path'],
        },
    )
    
    assert len(dataset) == 100
    # 3 PCs + 3 ancestries (K3) + 5 ancestries (K5) = 11 features
    assert dataset.data_array.shape == (100, 11)


def test_dataset_init_with_labels(temp_manifold_dir):
    """Test loading dataset with labels."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        label_column='Population',
    )
    
    assert len(dataset) == 100
    
    # Test get_labels
    labels = dataset.get_labels()
    assert len(labels) == 100
    assert all(label.startswith('Pop') for label in labels)
    
    # Test metadata includes labels
    sample = dataset[0]
    assert 'Population' in sample['metadata']
    assert 'latitude' in sample['metadata']
    assert 'longitude' in sample['metadata']


def test_dataset_with_colormap(temp_manifold_dir):
    """Test loading dataset with colormap."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        colormap_path=temp_manifold_dir['colormap_path'],
    )
    
    colormap = dataset.get_colormap()
    assert colormap is not None
    assert len(colormap) == 5
    assert 'Pop0' in colormap
    assert colormap['Pop0'] == '#FF0000'


def test_dataset_sample_id_alignment(temp_manifold_dir):
    """Test that sample IDs are properly aligned across data sources."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        admixture_paths={3: temp_manifold_dir['admix_k3_path']},
        labels_path=temp_manifold_dir['labels_path'],
    )
    
    sample_ids = dataset.get_sample_ids()
    assert len(sample_ids) == 100
    assert sample_ids[0] == 'sample_000'
    assert sample_ids[99] == 'sample_099'


def test_dataset_latitude_longitude_properties(temp_manifold_dir):
    """Test latitude and longitude properties."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
    )
    
    assert dataset.latitude is not None
    assert dataset.longitude is not None
    assert len(dataset.latitude) == 100
    assert len(dataset.longitude) == 100


def test_dataset_missing_sample_id_column(temp_manifold_dir):
    """Test error handling when sample_id column is missing."""
    # Create a CSV without sample_id
    bad_pca_path = temp_manifold_dir['dir'] / "bad_pca.csv"
    bad_pca_df = pd.DataFrame({
        'PC1': np.random.randn(10),
        'PC2': np.random.randn(10),
    })
    bad_pca_df.to_csv(bad_pca_path, index=False)
    
    with pytest.raises(ValueError, match="must contain 'sample_id' column"):
        ManifoldGeneticsDataset(pca_path=str(bad_pca_path))


def test_dataset_no_data_sources_error():
    """Test error when no data sources are provided."""
    with pytest.raises(ValueError, match="No data sources provided"):
        ManifoldGeneticsDataset()


def test_dataset_missing_label_column(temp_manifold_dir):
    """Test error when specified label column doesn't exist."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        label_column='NonexistentColumn',
    )
    
    with pytest.raises(ValueError, match="Label column 'NonexistentColumn' not found"):
        dataset.get_labels()


def test_dataset_partial_sample_overlap(temp_manifold_dir):
    """Test behavior when only some samples overlap between data sources."""
    # Create PCA with different sample IDs
    partial_pca_path = temp_manifold_dir['dir'] / "partial_pca.csv"
    partial_sample_ids = [f"sample_{i:03d}" for i in range(50, 150)]  # 50-149
    partial_pca_df = pd.DataFrame({
        'sample_id': partial_sample_ids,
        'PC1': np.random.randn(100),
        'PC2': np.random.randn(100),
    })
    partial_pca_df.to_csv(partial_pca_path, index=False)
    
    # Load with both PCA and labels (overlap is samples 50-99, so 50 samples)
    dataset = ManifoldGeneticsDataset(
        pca_path=str(partial_pca_path),
        labels_path=temp_manifold_dir['labels_path'],
    )
    
    # Should only have overlapping samples
    assert len(dataset) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
