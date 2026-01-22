"""
Tests for ManifoldGeneticsDataModule.

These tests verify the Lightning DataModule for manifold-genetics outputs.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from manylatents.popgen.data import ManifoldGeneticsDataModule


@pytest.fixture
def temp_manifold_split_data():
    """Create temporary directory with fit/transform split data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Training samples (fit)
        fit_sample_ids = [f"train_{i:03d}" for i in range(80)]
        fit_pca_data = {
            'sample_id': fit_sample_ids,
            'PC1': np.random.randn(80),
            'PC2': np.random.randn(80),
        }
        fit_pca_path = tmpdir / "fit_pca.csv"
        pd.DataFrame(fit_pca_data).to_csv(fit_pca_path, index=False)
        
        fit_admix_data = {
            'sample_id': fit_sample_ids,
            'Ancestry1': np.random.dirichlet([1, 1, 1], 80)[:, 0],
            'Ancestry2': np.random.dirichlet([1, 1, 1], 80)[:, 1],
            'Ancestry3': np.random.dirichlet([1, 1, 1], 80)[:, 2],
        }
        fit_admix_path = tmpdir / "fit.K3.csv"
        pd.DataFrame(fit_admix_data).to_csv(fit_admix_path, index=False)
        
        # Test samples (transform)
        transform_sample_ids = [f"test_{i:03d}" for i in range(20)]
        transform_pca_data = {
            'sample_id': transform_sample_ids,
            'PC1': np.random.randn(20),
            'PC2': np.random.randn(20),
        }
        transform_pca_path = tmpdir / "transform_pca.csv"
        pd.DataFrame(transform_pca_data).to_csv(transform_pca_path, index=False)
        
        transform_admix_data = {
            'sample_id': transform_sample_ids,
            'Ancestry1': np.random.dirichlet([1, 1, 1], 20)[:, 0],
            'Ancestry2': np.random.dirichlet([1, 1, 1], 20)[:, 1],
            'Ancestry3': np.random.dirichlet([1, 1, 1], 20)[:, 2],
        }
        transform_admix_path = tmpdir / "transform.K3.csv"
        pd.DataFrame(transform_admix_data).to_csv(transform_admix_path, index=False)
        
        # Labels for all samples
        all_sample_ids = fit_sample_ids + transform_sample_ids
        labels_data = {
            'sample_id': all_sample_ids,
            'Population': [f'Pop{i % 3}' for i in range(100)],
        }
        labels_path = tmpdir / "labels.csv"
        pd.DataFrame(labels_data).to_csv(labels_path, index=False)
        
        # Colormap
        colormap = {'Pop0': '#FF0000', 'Pop1': '#00FF00', 'Pop2': '#0000FF'}
        colormap_path = tmpdir / "colormap.json"
        with open(colormap_path, 'w') as f:
            json.dump(colormap, f)
        
        yield {
            'fit_pca_path': str(fit_pca_path),
            'transform_pca_path': str(transform_pca_path),
            'fit_admix_path': str(fit_admix_path),
            'transform_admix_path': str(transform_admix_path),
            'labels_path': str(labels_path),
            'colormap_path': str(colormap_path),
        }


def test_datamodule_init_split_mode(temp_manifold_split_data):
    """Test DataModule initialization in split mode."""
    datamodule = ManifoldGeneticsDataModule(
        batch_size=16,
        num_workers=0,
        mode='split',
        fit_pca_path=temp_manifold_split_data['fit_pca_path'],
        transform_pca_path=temp_manifold_split_data['transform_pca_path'],
        labels_path=temp_manifold_split_data['labels_path'],
        colormap_path=temp_manifold_split_data['colormap_path'],
    )
    
    assert datamodule.mode == 'split'
    assert datamodule.batch_size == 16


def test_datamodule_setup_split_mode(temp_manifold_split_data):
    """Test DataModule setup with train/test split."""
    datamodule = ManifoldGeneticsDataModule(
        batch_size=16,
        num_workers=0,
        mode='split',
        fit_pca_path=temp_manifold_split_data['fit_pca_path'],
        transform_pca_path=temp_manifold_split_data['transform_pca_path'],
        fit_admixture_paths={3: temp_manifold_split_data['fit_admix_path']},
        transform_admixture_paths={3: temp_manifold_split_data['transform_admix_path']},
        labels_path=temp_manifold_split_data['labels_path'],
        colormap_path=temp_manifold_split_data['colormap_path'],
    )
    
    datamodule.setup()
    
    # Check train dataset
    assert datamodule.train_dataset is not None
    assert len(datamodule.train_dataset) == 80  # fit samples
    
    # Check test dataset
    assert datamodule.test_dataset is not None
    assert len(datamodule.test_dataset) == 20  # transform samples
    
    # Verify they're different datasets
    assert datamodule.train_dataset is not datamodule.test_dataset


def test_datamodule_setup_full_mode(temp_manifold_split_data):
    """Test DataModule setup in full mode (same data for train and test)."""
    datamodule = ManifoldGeneticsDataModule(
        batch_size=16,
        num_workers=0,
        mode='full',
        transform_pca_path=temp_manifold_split_data['transform_pca_path'],
        labels_path=temp_manifold_split_data['labels_path'],
    )
    
    datamodule.setup()
    
    assert datamodule.train_dataset is not None
    assert datamodule.test_dataset is not None
    
    # In full mode, train and test should be the same object
    assert datamodule.train_dataset is datamodule.test_dataset
    assert len(datamodule.train_dataset) == 20


def test_datamodule_dataloaders(temp_manifold_split_data):
    """Test creating DataLoaders."""
    datamodule = ManifoldGeneticsDataModule(
        batch_size=8,
        num_workers=0,
        mode='split',
        fit_pca_path=temp_manifold_split_data['fit_pca_path'],
        transform_pca_path=temp_manifold_split_data['transform_pca_path'],
        labels_path=temp_manifold_split_data['labels_path'],
    )
    
    datamodule.setup()
    
    # Test train dataloader
    train_loader = datamodule.train_dataloader()
    assert train_loader.batch_size == 8
    
    # Test getting a batch
    batch = next(iter(train_loader))
    assert 'data' in batch
    assert 'metadata' in batch
    assert batch['data'].shape[0] <= 8  # Batch size
    assert isinstance(batch['data'], torch.Tensor)
    assert batch['data'].dtype == torch.float32
    
    # Test val dataloader
    val_loader = datamodule.val_dataloader()
    assert val_loader is not None
    
    # Test test dataloader
    test_loader = datamodule.test_dataloader()
    assert test_loader is not None
    test_batch = next(iter(test_loader))
    assert test_batch['data'].shape[0] <= 8


def test_datamodule_shuffle_train(temp_manifold_split_data):
    """Test that training data is shuffled when shuffle_traindata=True."""
    datamodule = ManifoldGeneticsDataModule(
        batch_size=8,
        num_workers=0,
        mode='split',
        fit_pca_path=temp_manifold_split_data['fit_pca_path'],
        transform_pca_path=temp_manifold_split_data['transform_pca_path'],
        labels_path=temp_manifold_split_data['labels_path'],
        shuffle_traindata=True,
    )
    
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    
    # The DataLoader should have shuffle=True
    # We can't directly check this, but we verify it was created successfully
    assert train_loader is not None


def test_datamodule_no_shuffle_train(temp_manifold_split_data):
    """Test that training data is not shuffled when shuffle_traindata=False."""
    datamodule = ManifoldGeneticsDataModule(
        batch_size=8,
        num_workers=0,
        mode='split',
        fit_pca_path=temp_manifold_split_data['fit_pca_path'],
        transform_pca_path=temp_manifold_split_data['transform_pca_path'],
        labels_path=temp_manifold_split_data['labels_path'],
        shuffle_traindata=False,
    )
    
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    assert train_loader is not None


def test_datamodule_invalid_mode(temp_manifold_split_data):
    """Test error handling for invalid mode."""
    datamodule = ManifoldGeneticsDataModule(
        batch_size=8,
        num_workers=0,
        mode='invalid_mode',
        fit_pca_path=temp_manifold_split_data['fit_pca_path'],
    )
    
    with pytest.raises(ValueError, match="Invalid mode"):
        datamodule.setup()


def test_datamodule_with_multiple_admixture_k(temp_manifold_split_data):
    """Test DataModule with multiple admixture K values."""
    # Create additional K=5 admixture files
    tmpdir = Path(temp_manifold_split_data['fit_pca_path']).parent
    
    fit_sample_ids = [f"train_{i:03d}" for i in range(80)]
    fit_admix_k5_data = {
        'sample_id': fit_sample_ids,
        **{f'Ancestry{i}': np.random.dirichlet([1]*5, 80)[:, i-1] for i in range(1, 6)}
    }
    fit_admix_k5_path = tmpdir / "fit.K5.csv"
    pd.DataFrame(fit_admix_k5_data).to_csv(fit_admix_k5_path, index=False)
    
    transform_sample_ids = [f"test_{i:03d}" for i in range(20)]
    transform_admix_k5_data = {
        'sample_id': transform_sample_ids,
        **{f'Ancestry{i}': np.random.dirichlet([1]*5, 20)[:, i-1] for i in range(1, 6)}
    }
    transform_admix_k5_path = tmpdir / "transform.K5.csv"
    pd.DataFrame(transform_admix_k5_data).to_csv(transform_admix_k5_path, index=False)
    
    datamodule = ManifoldGeneticsDataModule(
        batch_size=8,
        num_workers=0,
        mode='split',
        fit_pca_path=temp_manifold_split_data['fit_pca_path'],
        transform_pca_path=temp_manifold_split_data['transform_pca_path'],
        fit_admixture_paths={
            3: temp_manifold_split_data['fit_admix_path'],
            5: str(fit_admix_k5_path),
        },
        transform_admixture_paths={
            3: temp_manifold_split_data['transform_admix_path'],
            5: str(transform_admix_k5_path),
        },
        labels_path=temp_manifold_split_data['labels_path'],
    )
    
    datamodule.setup()
    
    # Train dataset should have 2 PCs + 3 ancestries (K3) + 5 ancestries (K5) = 10 features
    batch = next(iter(datamodule.train_dataloader()))
    assert batch['data'].shape[1] == 10


def test_datamodule_collate_fn(temp_manifold_split_data):
    """Test the collate function."""
    datamodule = ManifoldGeneticsDataModule(
        batch_size=4,
        num_workers=0,
        mode='split',
        fit_pca_path=temp_manifold_split_data['fit_pca_path'],
        transform_pca_path=temp_manifold_split_data['transform_pca_path'],
        labels_path=temp_manifold_split_data['labels_path'],
    )
    
    datamodule.setup()
    
    # Get a batch
    batch = next(iter(datamodule.train_dataloader()))
    
    # Check batch structure
    assert isinstance(batch, dict)
    assert 'data' in batch
    assert 'metadata' in batch
    
    # Check data tensor
    assert isinstance(batch['data'], torch.Tensor)
    assert batch['data'].dtype == torch.float32
    assert len(batch['data'].shape) == 2  # (batch_size, features)
    
    # Check metadata
    assert isinstance(batch['metadata'], list)
    assert len(batch['metadata']) == batch['data'].shape[0]
    assert all(isinstance(m, dict) for m in batch['metadata'])
    assert all('sample_id' in m for m in batch['metadata'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
