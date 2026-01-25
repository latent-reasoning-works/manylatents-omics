# ClinVar Pipeline Specification

## ADDED Requirements

### Requirement: ClinVar Data Module

The system SHALL provide a `ClinVarDataModule` that loads preprocessed ClinVar variant sequences for DNA, RNA, and Protein modalities.

#### Scenario: Load sequences by modality

- **WHEN** `ClinVarDataModule` is instantiated with a valid `data_dir`
- **THEN** `get_sequences()` returns a dict with keys `["dna", "rna", "protein"]`
- **AND** each value is a list of sequence strings aligned by variant index

#### Scenario: Filter by pathogenicity

- **WHEN** `ClinVarDataModule` is instantiated with `pathogenicity="pathogenic"`
- **THEN** `get_labels()` returns an array of all 1s (pathogenic only)
- **AND** `get_variant_ids()` returns only pathogenic variant IDs

### Requirement: Batch Encoder Algorithm

The system SHALL provide a `BatchEncoder` LatentModule that wraps foundation model encoders for batch sequence processing.

#### Scenario: Encode DNA with Evo2

- **WHEN** `BatchEncoder` is configured with `modality="dna"` and an Evo2 encoder
- **THEN** `transform()` encodes all DNA sequences from `datamodule.get_sequences()["dna"]`
- **AND** returns a tensor of shape `[N, 2048]` where N is the variant count

#### Scenario: Save embeddings to file

- **WHEN** `BatchEncoder` is configured with `save_path="embeddings/evo2.pt"`
- **THEN** embeddings are saved to the specified path after encoding
- **AND** the file contains `embeddings`, `variant_ids`, and `labels` keys

### Requirement: Multi-Channel Precomputed Data Module

The system SHALL extend `PrecomputedDataModule` to support loading multiple embedding channels from HDF5 or directory structure.

#### Scenario: Load multi-channel HDF5

- **WHEN** `PrecomputedDataModule` is instantiated with `channels=["dna/evo2", "protein/esm3"]`
- **THEN** `get_embeddings()` returns a dict mapping channel names to tensors
- **AND** all tensors have the same first dimension (aligned by variant)

#### Scenario: Backward compatibility

- **WHEN** `PrecomputedDataModule` is instantiated without `channels` parameter
- **THEN** it behaves as the existing single-channel implementation
- **AND** `get_tensor()` returns the single embedding tensor

### Requirement: Precomputed Fusion Module

The system SHALL provide a `PrecomputedFusionModule` LatentModule that fuses multi-channel embeddings from a precomputed data module.

#### Scenario: Concatenation fusion

- **WHEN** `PrecomputedFusionModule` is configured with `strategy="concat"`
- **THEN** `transform()` concatenates embeddings from all channels
- **AND** output dimension equals sum of input channel dimensions

#### Scenario: Weighted fusion

- **WHEN** `PrecomputedFusionModule` is configured with `strategy="weighted"` and `weights={"dna": 0.5, "protein": 0.5}`
- **THEN** embeddings are L2-normalized, scaled by weights, and summed
- **AND** output dimension equals the dimension of any single channel

#### Scenario: Selective channels

- **WHEN** `PrecomputedFusionModule` is configured with `channels=["dna", "protein"]`
- **THEN** only the specified channels are fused
- **AND** other channels from the datamodule are ignored

### Requirement: Hydra Experiment Configs

The system SHALL provide Hydra experiment configs for ClinVar encoding and analysis.

#### Scenario: DNA encoding config

- **WHEN** user runs `python -m manylatents.main +experiment=clinvar/encode_dna`
- **THEN** the pipeline encodes DNA sequences with Evo2
- **AND** saves embeddings to `${paths.output_dir}/embeddings/clinvar/evo2.pt`

#### Scenario: Geometric analysis config

- **WHEN** user runs `python -m manylatents.main +experiment=clinvar/geometric_analysis`
- **THEN** the pipeline loads precomputed embeddings
- **AND** computes PR, LID, TSA metrics on fused embeddings
- **AND** sweeps over n_components: 5, 50, 100

### Requirement: ClinVar Download Script

The system SHALL provide a `scripts/download_clinvar.py` script for one-time bulk download and preprocessing.

#### Scenario: Download and preprocess

- **WHEN** `download_clinvar.py` is executed
- **THEN** it downloads variant_summary from NCBI FTP
- **AND** fetches sequences from Ensembl REST API
- **AND** writes files to `data/clinvar/{dna,rna,protein}.fasta` and `variants.tsv`
