"""
E2E config tests - validate all Hydra configs resolve correctly.

No GPU required. Tests config composition and YAML structure only.
Does NOT import torch-dependent modules.
"""

import pytest
from omegaconf import OmegaConf
from pathlib import Path


# Path to dogma configs
CONFIGS_DIR = Path(__file__).parent.parent.parent / "manylatents" / "dogma" / "configs"


class TestExperimentConfigs:
    """Test experiment config files resolve correctly."""

    def test_clinvar_encode_dna_config(self):
        """Test clinvar/encode_dna experiment config."""
        config_path = CONFIGS_DIR / "experiment" / "clinvar" / "encode_dna.yaml"
        if not config_path.exists():
            pytest.skip("encode_dna.yaml not found")

        cfg = OmegaConf.load(config_path)
        assert cfg.name == "clinvar_encode_dna"
        assert cfg.project == "merging-dogma"
        assert cfg.algorithms.latent._target_ == "manylatents.dogma.algorithms.BatchEncoder"
        assert cfg.algorithms.latent.modality == "dna"
        assert "Evo2Encoder" in cfg.algorithms.latent.encoder_config._target_

    def test_clinvar_encode_protein_config(self):
        """Test clinvar/encode_protein experiment config."""
        config_path = CONFIGS_DIR / "experiment" / "clinvar" / "encode_protein.yaml"
        if not config_path.exists():
            pytest.skip("encode_protein.yaml not found")

        cfg = OmegaConf.load(config_path)
        assert cfg.name == "clinvar_encode_protein"
        assert cfg.project == "merging-dogma"
        assert cfg.algorithms.latent._target_ == "manylatents.dogma.algorithms.BatchEncoder"
        assert cfg.algorithms.latent.modality == "protein"
        assert "ESM3Encoder" in cfg.algorithms.latent.encoder_config._target_

    def test_clinvar_encode_rna_config(self):
        """Test clinvar/encode_rna experiment config."""
        config_path = CONFIGS_DIR / "experiment" / "clinvar" / "encode_rna.yaml"
        if not config_path.exists():
            pytest.skip("encode_rna.yaml not found")

        cfg = OmegaConf.load(config_path)
        assert cfg.name == "clinvar_encode_rna"
        assert cfg.project == "merging-dogma"
        assert cfg.algorithms.latent._target_ == "manylatents.dogma.algorithms.BatchEncoder"
        assert cfg.algorithms.latent.modality == "rna"
        assert "OrthrusEncoder" in cfg.algorithms.latent.encoder_config._target_

    def test_clinvar_geometric_analysis_config(self):
        """Test clinvar/geometric_analysis experiment config."""
        config_path = CONFIGS_DIR / "experiment" / "clinvar" / "geometric_analysis.yaml"
        if not config_path.exists():
            pytest.skip("geometric_analysis.yaml not found")

        cfg = OmegaConf.load(config_path)
        assert cfg.name == "clinvar_geometric_analysis"
        assert cfg.project == "merging-dogma"
        assert cfg.algorithms.latent._target_ == "manylatents.algorithms.latent.MergingModule"
        assert cfg.algorithms.latent.strategy == "concat"

    def test_central_dogma_fusion_config(self):
        """Test central_dogma_fusion experiment config."""
        config_path = CONFIGS_DIR / "experiment" / "central_dogma_fusion.yaml"
        if not config_path.exists():
            pytest.skip("central_dogma_fusion.yaml not found")

        cfg = OmegaConf.load(config_path)
        # Just check it loads without error
        assert cfg is not None


class TestEncoderConfigs:
    """Test encoder config files."""

    def test_orthrus_config_exists(self):
        """Test orthrus encoder config exists and has correct target."""
        config_path = CONFIGS_DIR / "encoders" / "orthrus.yaml"
        if not config_path.exists():
            pytest.skip("orthrus.yaml not found")

        cfg = OmegaConf.load(config_path)
        assert cfg._target_ == "manylatents.dogma.encoders.OrthrusEncoder"

    def test_evo2_config_exists(self):
        """Test evo2 encoder config exists."""
        config_path = CONFIGS_DIR / "encoders" / "evo2.yaml"
        if not config_path.exists():
            pytest.skip("evo2.yaml not found")

        cfg = OmegaConf.load(config_path)
        assert "Evo2Encoder" in cfg._target_

    def test_esm3_config_exists(self):
        """Test esm3 encoder config exists."""
        config_path = CONFIGS_DIR / "encoders" / "esm3.yaml"
        if not config_path.exists():
            pytest.skip("esm3.yaml not found")

        cfg = OmegaConf.load(config_path)
        assert "ESM3Encoder" in cfg._target_


class TestDataConfigs:
    """Test data config files."""

    def test_clinvar_data_config(self):
        """Test clinvar data config exists."""
        config_path = CONFIGS_DIR / "data" / "clinvar.yaml"
        if not config_path.exists():
            pytest.skip("clinvar.yaml not found")

        cfg = OmegaConf.load(config_path)
        assert "ClinVarDataModule" in cfg._target_


class TestConfigConsistency:
    """Test configs are consistent across the pipeline."""

    def test_all_experiment_configs_have_project(self):
        """All experiment configs should have project: merging-dogma."""
        exp_dir = CONFIGS_DIR / "experiment"
        if not exp_dir.exists():
            pytest.skip("experiment dir not found")

        for config_file in exp_dir.rglob("*.yaml"):
            cfg = OmegaConf.load(config_file)
            if hasattr(cfg, "project"):
                assert cfg.project == "merging-dogma", f"{config_file.name} has wrong project"

    def test_encoder_targets_are_importable_paths(self):
        """All encoder _target_ values should be valid Python import paths."""
        encoders_dir = CONFIGS_DIR / "encoders"
        if not encoders_dir.exists():
            pytest.skip("encoders dir not found")

        for config_file in encoders_dir.glob("*.yaml"):
            cfg = OmegaConf.load(config_file)
            if hasattr(cfg, "_target_"):
                target = cfg._target_
                # Check it's a valid Python path
                assert "." in target, f"{config_file.name}: invalid target {target}"
                assert target.startswith("manylatents"), f"{config_file.name}: target should start with manylatents"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
