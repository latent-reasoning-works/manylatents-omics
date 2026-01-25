# Workshop Paper: Geometric Structure in Cross-Dogma Embeddings

## Experiment Specification

### Goal
Compare fusion strategies for combining DNA (Evo2) and Protein (ESM3) embeddings on ClinVar pathogenic/benign variant classification.

### Metrics
- **Intrinsic dimensionality**: LID, PR, TSA (k = 5, 10, 25, 50, 100)
- **Downstream classification**: Logistic regression, MLP (5-fold CV, AUROC/AUPRC/Accuracy)

---

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Download    │────▶│  2. Encode      │────▶│  3. Fuse +      │
│     ClinVar     │     │     DNA/Protein │     │     Evaluate    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
     CPU only            GPU (Ampere+)           GPU or CPU
     ~5 min              ~30 min each            ~10 min each
```

---

## Step 1: Download ClinVar Data

**Script**: `scripts/download_clinvar.py`

**What it does**:
- Downloads variant_summary from NCBI FTP
- Filters to BRCA1/BRCA2 pathogenic/benign variants
- Fetches DNA sequences from Ensembl REST API (±50bp context)
- Translates to protein sequences

**Output**:
```
data/clinvar/
├── variants.csv          # Variant metadata + labels
├── dna_sequences.fasta   # DNA sequences for Evo2
└── protein_sequences.fasta  # Protein sequences for ESM3
```

**Command**:
```bash
python scripts/download_clinvar.py --genes BRCA1,BRCA2 --output data/clinvar/
```

**Verification**:
```bash
wc -l data/clinvar/variants.csv
# Expected: 500-2000 variants
```

---

## Step 2: Encode with Foundation Models

### 2a. DNA Encoding (Evo2)

**Requirements**: L40S/A100/H100 GPU, ~24GB VRAM

**Config**: `experiment=clinvar/encode_dna`

**Command**:
```bash
# Interactive GPU session
salloc --gpus=l40s:1 --time=2:00:00
module load cuda/12.4.1

# Run encoding
python -m manylatents.omics.main --config-name=config experiment=clinvar/encode_dna
```

**Output**:
```
outputs/embeddings/clinvar/
├── evo2.pt               # (n_variants, 1920) DNA embeddings
└── evo2_metadata.json    # Encoding config
```

### 2b. Protein Encoding (ESM3)

**Requirements**: GPU with 16GB+ VRAM

**Config**: `experiment=clinvar/encode_protein`

**Command**:
```bash
python -m manylatents.omics.main --config-name=config experiment=clinvar/encode_protein
```

**Output**:
```
outputs/embeddings/clinvar/
├── esm3.pt               # (n_variants, 1536) Protein embeddings
└── esm3_metadata.json    # Encoding config
```

### 2c. Labels

**Generated during encoding** (or manually):
```
outputs/embeddings/clinvar/
└── labels.pt             # (n_variants,) 0=benign, 1=pathogenic
```

**Verification**:
```bash
python -c "
import torch
evo2 = torch.load('outputs/embeddings/clinvar/evo2.pt')
esm3 = torch.load('outputs/embeddings/clinvar/esm3.pt')
labels = torch.load('outputs/embeddings/clinvar/labels.pt')
print(f'DNA: {evo2.shape}, Protein: {esm3.shape}, Labels: {labels.shape}')
assert evo2.shape[0] == esm3.shape[0] == labels.shape[0]
print('All shapes match!')
"
```

---

## Step 3: Fusion Experiments

### Strategies to Compare

| Strategy | Config | Description | Output Dim |
|----------|--------|-------------|------------|
| concat | `fusion/concat` | Simple concatenation | 3456 (1920+1536) |
| concat_pca | `fusion/concat_pca` | Concat → PCA | 256 |
| modality_proj | `fusion/modality_proj` | Per-channel PCA → concat | 512 (256×2) |
| svd | `fusion/svd` | Concat → truncated SVD | 256 |
| autoencoder | `fusion/autoencoder` | Learned bottleneck | 256 |
| frobenius_ae | `fusion/frobenius_ae` | Jacobian-regularized AE | 256 |

### Single Strategy Run

```bash
python -m manylatents.omics.main --config-name=config experiment=fusion/concat_pca
```

### Full Sweep (All Linear Strategies)

```bash
python -m manylatents.omics.main -m --config-name=config \
  experiment=fusion/concat,fusion/concat_pca,fusion/modality_proj,fusion/svd
```

### Full Sweep (Including Learned)

```bash
python -m manylatents.omics.main -m --config-name=config \
  experiment=fusion/concat,fusion/concat_pca,fusion/modality_proj,fusion/svd,fusion/autoencoder,fusion/frobenius_ae
```

### Output per Strategy

```
outputs/fusion/
├── embeddings_fusion_concat_pca_20260124_120000.npy
├── metrics_summary_fusion_concat_pca_20260124_120000.csv
└── metrics_per_sample_fusion_concat_pca_20260124_120000.csv
```

**metrics_summary.csv columns**:
```
embedding.lid__k_5, embedding.lid__k_10, ..., embedding.lid__k_100,
embedding.pr__n_neighbors_5, ..., embedding.pr__n_neighbors_100,
embedding.tsa__n_neighbors_5, ..., embedding.tsa__n_neighbors_100,
logistic_accuracy, logistic_auroc, logistic_auprc,
mlp_accuracy, mlp_auroc, mlp_auprc
```

---

## Cluster Submission

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=fusion-sweep
#SBATCH --output=logs/fusion-%j.out
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

module load cuda/12.4.1

# Run sweep
python -m manylatents.omics.main -m --config-name=config \
  experiment=fusion/concat,fusion/concat_pca,fusion/modality_proj,fusion/svd
```

### Via Hydra Launcher (if configured)

```bash
python -m manylatents.omics.main -m --config-name=config \
  cluster=mila_remote \
  resources=gpu \
  experiment=fusion/concat,fusion/concat_pca,fusion/modality_proj,fusion/svd
```

---

## Expected Results

### Hypotheses

1. **concat_pca** and **svd** should have similar LID (both linear projections)
2. **modality_proj** preserves per-channel structure better (higher PR per modality)
3. **Learned methods** (autoencoder, frobenius_ae) may achieve lower reconstruction but similar classification
4. **Classification**: All methods should achieve AUROC > 0.7 if embeddings capture pathogenicity signal

### Analysis

After experiments complete:
```bash
# Aggregate results
python scripts/aggregate_fusion_results.py outputs/fusion/

# Output: fusion_comparison.csv with all strategies × metrics
```

---

## Troubleshooting

### Missing CUDA
```bash
module load cuda/12.4.1
```

### OOM on Evo2
```bash
# Reduce batch size
python -m manylatents.omics.main --config-name=config experiment=clinvar/encode_dna \
  data.batch_size=4
```

### No labels in embeddings
```bash
# Check labels exist
ls outputs/embeddings/clinvar/labels.pt

# If missing, generate from variants.csv
python -c "
import pandas as pd
import torch
df = pd.read_csv('data/clinvar/variants.csv')
labels = torch.tensor((df['clinical_significance'] == 'Pathogenic').astype(int).values)
torch.save(labels, 'outputs/embeddings/clinvar/labels.pt')
"
```

---

## File Checklist

Before running fusion experiments, verify:

- [ ] `data/clinvar/variants.csv` exists
- [ ] `data/clinvar/dna_sequences.fasta` exists
- [ ] `data/clinvar/protein_sequences.fasta` exists
- [ ] `outputs/embeddings/clinvar/evo2.pt` exists
- [ ] `outputs/embeddings/clinvar/esm3.pt` exists
- [ ] `outputs/embeddings/clinvar/labels.pt` exists
- [ ] All embedding shapes match (same n_variants)
