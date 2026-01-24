# ClinVar Central Dogma Analysis Pipeline

Complete reproducible pipeline for geometric analysis of foundation model embeddings across the central dogma using ClinVar variants.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA PREPARATION                                                    │
│ Entry: ClinVar VCF + RefSeq                                                  │
│ Exit:  data/clinvar/{variants.tsv, dna.fasta, rna.fasta, protein.fasta}     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: EMBEDDING GENERATION (All in same environment)                      │
│ Entry: FASTA files                                                           │
│ Exit:  embeddings/clinvar/{evo2,orthrus,esm3}.pt                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: FUSION & GEOMETRIC ANALYSIS                                         │
│ Entry: Per-modality embeddings                                               │
│ Exit:  Fused embeddings + geometric metrics                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Encoders

All three encoders work in the **same environment** (mamba-ssm 2.x compatible):

| Encoder | Modality | Embedding Dim | HuggingFace Model |
|---------|----------|---------------|-------------------|
| **Evo2Encoder** | DNA | 1920 | arcinstitute/evo2_1b_base |
| **OrthrusEncoder** | RNA | 256 | quietflamingo/orthrus-base-4-track |
| **ESM3Encoder** | Protein | 1536 | esm3_sm_open_v1 |

**Note**: OrthrusEncoder was re-implemented to use mamba-ssm 2.x Block API, eliminating
the need for a separate environment. See `manylatents/dogma/encoders/orthrus_native.py`.

---

## Quick Start

### E2E Test (3-way fusion)

```bash
# Submit GPU job
sbatch scripts/run_e2e_test.sh

# Check results
cat logs/clinvar_e2e_*.out
```

### Hydra Experiments

```bash
# Encode DNA
python -m manylatents.main --config-name=config experiment=clinvar/encode_dna

# Encode RNA
python -m manylatents.main --config-name=config experiment=clinvar/encode_rna

# Encode Protein
python -m manylatents.main --config-name=config experiment=clinvar/encode_protein

# Geometric analysis on fused embeddings
python -m manylatents.main --config-name=config experiment=clinvar/geometric_analysis
```

---

## Stage 1: Data Preparation

### Entry Point
- ClinVar VCF from NCBI: `ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/`

### Script

```bash
python scripts/download_clinvar.py \
    --genes BRCA1,BRCA2 \
    --output data/clinvar/
```

### Exit Points
| File | Description |
|------|-------------|
| `variants.tsv` | Variant metadata |
| `dna.fasta` | DNA context sequences |
| `protein.fasta` | Translated protein sequences |
| `labels.csv` | Pathogenicity labels |

---

## Stage 2: Embedding Generation

All encoders run in the same GPU job:

```bash
# Via Hydra (recommended)
python -m manylatents.main --config-name=config experiment=clinvar/encode_dna
python -m manylatents.main --config-name=config experiment=clinvar/encode_rna
python -m manylatents.main --config-name=config experiment=clinvar/encode_protein

# Or parallel SLURM
sbatch scripts/run_e2e_test.sh
```

### Exit Points
| File | Encoder | Dimensions |
|------|---------|------------|
| `embeddings/clinvar/evo2.pt` | Evo2 | (N, 1920) |
| `embeddings/clinvar/orthrus.pt` | Orthrus | (N, 256) |
| `embeddings/clinvar/esm3.pt` | ESM3 | (N, 1536) |

---

## Stage 3: Fusion & Geometric Analysis

### MergingModule Strategies

| Strategy | Output Dim | Description |
|----------|------------|-------------|
| `concat` | 3712 | Concatenation [DNA; RNA; Protein] |
| `mean` | requires same dim | Element-wise mean |
| `weighted_sum` | requires same dim | Weighted combination |

### Hydra Config

```bash
python -m manylatents.main --config-name=config experiment=clinvar/geometric_analysis

# With weighted fusion
python -m manylatents.main --config-name=config experiment=clinvar/geometric_analysis \
    algorithms.latent.strategy=weighted_sum \
    'algorithms.latent.weights={evo2: 0.5, esm3: 0.5}'
```

### Metrics

| Metric | Description |
|--------|-------------|
| `ParticipationRatio` | Effective dimensionality |
| `LocalIntrinsicDimensionality` | KNN-based local dimension |
| `TangentSpaceApproximation` | PCA-based local dimension |

---

## Directory Structure

```
omics/
├── manylatents/dogma/
│   ├── encoders/
│   │   ├── evo2.py              # DNA encoder
│   │   ├── orthrus_native.py    # RNA encoder (mamba-ssm 2.x)
│   │   └── esm3.py              # Protein encoder
│   ├── configs/experiment/clinvar/
│   │   ├── encode_dna.yaml
│   │   ├── encode_rna.yaml
│   │   ├── encode_protein.yaml
│   │   └── geometric_analysis.yaml
│   └── data/
│       └── clinvar_dataset.py   # ClinVarDataModule
├── scripts/
│   ├── download_clinvar.py      # Data preparation
│   ├── test_clinvar_e2e.py      # E2E test
│   └── run_e2e_test.sh          # SLURM submission
└── tests/dogma/
    └── test_config_e2e.py       # Config validation
```

---

## Environment Setup

Single environment for all encoders:

```bash
cd /network/scratch/c/cesar.valdez/lrw/omics

# Install wheelnext uv (for CUDA wheel variants)
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh

# Sync with dogma extras
uv sync --extra dogma --index-strategy unsafe-best-match

# Verify imports
uv run python -c "from manylatents.dogma.encoders import Evo2Encoder, OrthrusEncoder, ESM3Encoder; print('OK')"
```

---

## WandB Project

All experiments log to: **merging-dogma**

Example runs:
- E2E 3-way fusion: https://wandb.ai/cesar-valdez-mcgill-university/merging-dogma/runs/imnb1zu7

---

## Reproducibility Checklist

- [ ] ClinVar VCF version documented
- [ ] GPU type documented (L40S/H100 for Evo2)
- [ ] All embeddings saved with labels
- [ ] WandB run URL logged
- [ ] Config E2E tests pass (`uv run python -m pytest tests/dogma/test_config_e2e.py`)
