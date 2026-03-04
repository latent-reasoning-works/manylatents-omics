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
        
        # Create PCA CSV with correct column names (dim_1, dim_2, etc.)
        pca_data = {
            'sample_id': sample_ids,
            'dim_1': np.random.randn(100),
            'dim_2': np.random.randn(100),
            'dim_3': np.random.randn(100),
        }
        pca_df = pd.DataFrame(pca_data)
        pca_path = tmpdir / "pca.csv"
        pca_df.to_csv(pca_path, index=False)
        
        # Create Admixture CSVs for K=3 and K=5 (use component_i format)
        admix_k3_data = {
            'sample_id': sample_ids,
            'component_1': np.random.dirichlet([1, 1, 1], 100)[:, 0],
            'component_2': np.random.dirichlet([1, 1, 1], 100)[:, 1],
            'component_3': np.random.dirichlet([1, 1, 1], 100)[:, 2],
        }
        admix_k3_df = pd.DataFrame(admix_k3_data)
        admix_k3_path = tmpdir / "admix_k3.csv"
        admix_k3_df.to_csv(admix_k3_path, index=False)
        
        admix_k5_data = {
            'sample_id': sample_ids,
            **{f'component_{i}': np.random.dirichlet([1]*5, 100)[:, i-1] for i in range(1, 6)}
        }
        admix_k5_df = pd.DataFrame(admix_k5_data)
        admix_k5_path = tmpdir / "admix_k5.csv"
        admix_k5_df.to_csv(admix_k5_path, index=False)
        
        # Create Labels CSV
        labels_data = {
            'sample_id': sample_ids,
            'Population': [f'Pop{i % 5}' for i in range(100)],
            'Genetic_region': [f'Region{i % 3}' for i in range(100)],
            'latitude': np.random.uniform(-90, 90, 100),
            'longitude': np.random.uniform(-180, 180, 100),
        }
        labels_df = pd.DataFrame(labels_data)
        labels_path = tmpdir / "labels.csv"
        labels_df.to_csv(labels_path, index=False)
        
        # Create colormap JSON (nested by label type)
        colormap = {
            'Population': {
                'Pop0': '#FF0000',
                'Pop1': '#00FF00',
                'Pop2': '#0000FF',
                'Pop3': '#FFFF00',
                'Pop4': '#FF00FF',
            },
            'Genetic_region': {
                'Region0': '#FF6B6B',
                'Region1': '#4ECDC4',
                'Region2': '#95E1D3',
            }
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
    # Data array contains ONLY PCA features (input data)
    # Admixture is stored separately for metrics, NOT part of input features
    assert dataset.data_array.shape == (100, 3)  # 3 PCs only

    # Verify admixture stored separately
    assert 3 in dataset.admixture_ratios
    assert 5 in dataset.admixture_ratios
    assert len(dataset.admixture_ratios[3]) == 100
    assert len(dataset.admixture_ratios[5]) == 100
    # Verify named columns preserved
    assert 'sample_id' in dataset.admixture_ratios[3].columns
    assert 'component_1' in dataset.admixture_ratios[3].columns


def test_dataset_init_with_labels(temp_manifold_dir):
    """Test loading dataset with labels."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        label_column='Population',
    )

    assert len(dataset) == 100

    # Test get_labels returns integer-encoded labels (for torch.tensor compatibility)
    labels = dataset.get_labels()
    assert len(labels) == 100
    assert labels.dtype == np.int64
    assert set(labels) == {0, 1, 2, 3, 4}  # 5 unique populations (Pop0-Pop4)

    # Test get_label_names returns original string labels
    label_names = dataset.get_label_names()
    assert len(label_names) == 100
    assert all(label.startswith('Pop') for label in label_names)

    # Test get_label_classes returns unique labels in sorted order
    classes = dataset.get_label_classes()
    assert list(classes) == ['Pop0', 'Pop1', 'Pop2', 'Pop3', 'Pop4']

    # Verify encoding consistency: classes[i] corresponds to integer i
    for i, cls in enumerate(classes):
        mask = label_names == cls
        assert all(labels[mask] == i)

    # Test metadata includes labels
    sample = dataset[0]
    assert 'Population' in sample['metadata']
    assert 'latitude' in sample['metadata']
    assert 'longitude' in sample['metadata']


def test_dataset_with_colormap(temp_manifold_dir):
    """Test loading dataset with nested colormap."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        colormap_path=temp_manifold_dir['colormap_path'],
        label_column='Population',
    )
    
    colormap = dataset.get_colormap()
    assert colormap is not None
    assert len(colormap) == 5  # 5 populations
    assert 'Pop0' in colormap
    assert colormap['Pop0'] == '#FF0000'
    
    # Test with different label column
    dataset2 = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        colormap_path=temp_manifold_dir['colormap_path'],
        label_column='Genetic_region',
    )
    
    colormap2 = dataset2.get_colormap()
    assert colormap2 is not None
    assert len(colormap2) == 3  # 3 regions
    assert 'Region0' in colormap2
    assert colormap2['Region0'] == '#FF6B6B'


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
    # Create a CSV without sample_id (using dim_1, dim_2 for consistency)
    bad_pca_path = temp_manifold_dir['dir'] / "bad_pca.csv"
    bad_pca_df = pd.DataFrame({
        'dim_1': np.random.randn(10),
        'dim_2': np.random.randn(10),
    })
    bad_pca_df.to_csv(bad_pca_path, index=False)
    
    with pytest.raises(ValueError, match="must contain 'sample_id' column"):
        ManifoldGeneticsDataset(pca_path=str(bad_pca_path))


def test_dataset_no_data_sources_error():
    """Test error when no data sources are provided."""
    with pytest.raises(ValueError, match="No input data provided"):
        ManifoldGeneticsDataset()


def test_dataset_missing_label_column(temp_manifold_dir):
    """Test error when specified label column doesn't exist."""
    # Error is raised during initialization when label_column is validated
    with pytest.raises(ValueError, match="Label column 'NonexistentColumn' not found"):
        ManifoldGeneticsDataset(
            pca_path=temp_manifold_dir['pca_path'],
            labels_path=temp_manifold_dir['labels_path'],
            label_column='NonexistentColumn',
        )


def test_dataset_partial_sample_overlap(temp_manifold_dir):
    """Test behavior when only some samples overlap between data sources."""
    # Create PCA with different sample IDs (using dim_1, dim_2 columns)
    partial_pca_path = temp_manifold_dir['dir'] / "partial_pca.csv"
    partial_sample_ids = [f"sample_{i:03d}" for i in range(50, 150)]  # 50-149
    partial_pca_df = pd.DataFrame({
        'sample_id': partial_sample_ids,
        'dim_1': np.random.randn(100),
        'dim_2': np.random.randn(100),
    })
    partial_pca_df.to_csv(partial_pca_path, index=False)
    
    # Load with both PCA and labels (overlap is samples 50-99, so 50 samples)
    dataset = ManifoldGeneticsDataset(
        pca_path=str(partial_pca_path),
        labels_path=temp_manifold_dir['labels_path'],
    )
    
    # Should only have overlapping samples
    assert len(dataset) == 50


def test_get_colormap_info_integer_keys_align_with_labels(temp_manifold_dir):
    """Integer keys in cmap and label_names must align with get_labels() / get_label_classes()."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        colormap_path=temp_manifold_dir['colormap_path'],
        label_column='Population',
    )

    info = dataset.get_colormap_info()

    assert isinstance(info.cmap, dict)
    assert isinstance(info.label_names, dict)
    assert info.is_categorical is True

    classes = dataset.get_label_classes()  # sorted array of string labels
    labels = dataset.get_labels()          # integer-encoded

    # Keys must exactly match the set of integer encodings
    assert set(info.cmap.keys()) == set(range(len(classes)))
    assert set(info.label_names.keys()) == set(range(len(classes)))

    # label_names[i] must equal the string class for integer i
    for i, cls in enumerate(classes):
        assert info.label_names[i] == cls

    # Every integer in get_labels() must have an entry in cmap
    for encoded_int in labels:
        assert encoded_int in info.cmap


def test_get_colormap_info_colors_from_nested_colormap(temp_manifold_dir):
    """Colors must come from the correct sub-dict for each label_column."""
    colormap_fixture = {
        'Population': {
            'Pop0': '#FF0000', 'Pop1': '#00FF00', 'Pop2': '#0000FF',
            'Pop3': '#FFFF00', 'Pop4': '#FF00FF',
        },
        'Genetic_region': {
            'Region0': '#FF6B6B', 'Region1': '#4ECDC4', 'Region2': '#95E1D3',
        },
    }

    # Population column
    ds_pop = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        colormap_path=temp_manifold_dir['colormap_path'],
        label_column='Population',
    )
    info_pop = ds_pop.get_colormap_info()
    classes_pop = ds_pop.get_label_classes()
    for i, cls in enumerate(classes_pop):
        assert info_pop.cmap[i] == colormap_fixture['Population'][cls]

    # Genetic_region column
    ds_region = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        colormap_path=temp_manifold_dir['colormap_path'],
        label_column='Genetic_region',
    )
    info_region = ds_region.get_colormap_info()
    classes_region = ds_region.get_label_classes()
    for i, cls in enumerate(classes_region):
        assert info_region.cmap[i] == colormap_fixture['Genetic_region'][cls]

    # The two infos must be independent (different number of classes)
    assert len(info_pop.cmap) == 5
    assert len(info_region.cmap) == 3


def test_get_colormap_info_fallback_no_colormap(temp_manifold_dir):
    """Falls back to viridis string when no colormap is provided."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        label_column='Population',
    )

    info = dataset.get_colormap_info()

    assert info.cmap == "viridis"
    assert info.label_names is None
    assert info.is_categorical is True


def test_get_colormap_info_fallback_no_labels(temp_manifold_dir):
    """Falls back to viridis string when no labels CSV is provided."""
    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        colormap_path=temp_manifold_dir['colormap_path'],
    )

    info = dataset.get_colormap_info()

    assert info.cmap == "viridis"
    assert info.label_names is None


def test_get_colormap_info_missing_label_in_colormap(temp_manifold_dir):
    """Labels absent from the colormap JSON receive the grey fallback color (#808080)."""
    # Write a partial colormap that is missing Pop3 and Pop4
    partial_colormap = {
        'Population': {
            'Pop0': '#FF0000',
            'Pop1': '#00FF00',
            'Pop2': '#0000FF',
        }
    }
    partial_colormap_path = temp_manifold_dir['dir'] / "partial_colormap.json"
    with open(partial_colormap_path, 'w') as f:
        json.dump(partial_colormap, f)

    dataset = ManifoldGeneticsDataset(
        pca_path=temp_manifold_dir['pca_path'],
        labels_path=temp_manifold_dir['labels_path'],
        colormap_path=str(partial_colormap_path),
        label_column='Population',
    )

    info = dataset.get_colormap_info()
    classes = dataset.get_label_classes()

    # All classes must have an entry
    assert set(info.cmap.keys()) == set(range(len(classes)))

    for i, cls in enumerate(classes):
        if cls in partial_colormap['Population']:
            assert info.cmap[i] == partial_colormap['Population'][cls]
        else:
            assert info.cmap[i] == "#808080"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
