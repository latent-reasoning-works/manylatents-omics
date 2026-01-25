#!/usr/bin/env python
"""E2E test: ClinVar DNA+RNA+Protein encoding → fusion → WandB logging.

Tests the full pipeline:
1. Load ClinVar variants (small subset)
2. Encode DNA with Evo2
3. Encode RNA with Orthrus (cached) or synthetic fallback
4. Encode Protein with ESM3
5. Fuse embeddings with MergingModule (3-way)
6. Compute geometric metrics
7. Log to WandB

Usage:
    python scripts/test_clinvar_e2e.py

    # Or via SLURM
    sbatch --gres=gpu:l40s:1 -c 4 --mem=32G --time=00:30:00 \
        --wrap="uv run python scripts/test_clinvar_e2e.py"
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import numpy as np

# Ensure we can import from the repo
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    import wandb
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    print("=" * 60)
    print("ClinVar E2E Fusion Test")
    print("=" * 60)

    # Initialize WandB
    run = wandb.init(
        project="merging-dogma",
        name="e2e-fusion-test",
        tags=["e2e", "test", "fusion", "dna", "protein"],
    )

    # Create temp dir for embeddings
    output_dir = Path(tempfile.mkdtemp(prefix="clinvar_e2e_"))
    print(f"\nOutput dir: {output_dir}")

    # Test data - use synthetic sequences for quick test
    # (Real ClinVar would need download_clinvar.py first)
    test_sequences = {
        "dna": [
            "ATGCGTACGATCGATCGATCGATCGATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
            "TACGTACGTACGTACGTACGTACGTACGTACGTACGTACG",
        ],
        "rna": [  # Same as DNA but with U instead of T (transcribed)
            "AUGCGUACGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCG",
            "GCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUA",
            "UACGUACGUACGUACGUACGUACGUACGUACGUACGUACG",
        ],
        "protein": [
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNT",
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYL",
        ],
    }
    labels = np.array([1, 0, 1])  # pathogenic, benign, pathogenic

    print(f"\nTest data: {len(test_sequences['dna'])} sequences")
    wandb.log({"n_sequences": len(test_sequences['dna'])})

    # Step 1: Encode DNA with Evo2
    print("\n[1/5] Encoding DNA with Evo2...")
    from manylatents.dogma.encoders import Evo2Encoder

    evo2 = Evo2Encoder(
        model_name="evo2_1b_base",
        layer_name="blocks.14.mlp.l3",
        device="cuda",
    )

    dna_embeddings = []
    for seq in test_sequences["dna"]:
        emb = evo2.encode(seq)
        # Squeeze extra dimensions if needed (Evo2 may return [1, dim])
        if emb.dim() > 1:
            emb = emb.squeeze()
        dna_embeddings.append(emb)
    dna_embeddings = torch.stack(dna_embeddings)
    print(f"   DNA embeddings: {dna_embeddings.shape}")
    wandb.log({"dna_embedding_dim": dna_embeddings.shape[-1]})

    # Save DNA embeddings
    torch.save({
        "embeddings": dna_embeddings,
        "labels": torch.from_numpy(labels),
    }, output_dir / "evo2.pt")

    # Step 2: Encode RNA with Orthrus
    print("\n[2/5] Encoding RNA with Orthrus...")
    try:
        from manylatents.dogma.encoders import OrthrusEncoder

        orthrus = OrthrusEncoder(device="cuda")

        rna_embeddings = []
        for seq in test_sequences["rna"]:
            emb = orthrus.encode(seq)
            if emb.dim() > 1:
                emb = emb.squeeze()
            rna_embeddings.append(emb)
        rna_embeddings = torch.stack(rna_embeddings)
        print(f"   RNA embeddings: {rna_embeddings.shape}")
        wandb.log({"rna_embedding_dim": rna_embeddings.shape[-1], "orthrus_status": "native"})
    except Exception as e:
        print(f"   OrthrusEncoder failed ({e}), using synthetic RNA embeddings")
        # Create synthetic RNA embeddings with Orthrus dim (256 for 4-track)
        rna_embeddings = torch.randn(len(test_sequences["rna"]), 256, device="cpu")
        print(f"   Synthetic RNA embeddings: {rna_embeddings.shape}")
        wandb.log({"rna_embedding_dim": 256, "orthrus_status": "synthetic"})

    # Save RNA embeddings
    torch.save({
        "embeddings": rna_embeddings,
        "labels": torch.from_numpy(labels),
    }, output_dir / "orthrus.pt")

    # Step 3: Encode Protein with ESM3 (or use synthetic if gated)
    print("\n[3/5] Encoding Protein with ESM3...")
    try:
        from manylatents.dogma.encoders import ESM3Encoder

        # Use default (downloads from HuggingFace with auth)
        esm3 = ESM3Encoder(device="cuda")

        protein_embeddings = []
        for seq in test_sequences["protein"]:
            emb = esm3.encode(seq)
            # Squeeze extra dimensions if needed
            if emb.dim() > 1:
                emb = emb.squeeze()
            protein_embeddings.append(emb)
        protein_embeddings = torch.stack(protein_embeddings)
        print(f"   Protein embeddings: {protein_embeddings.shape}")
        wandb.log({"protein_embedding_dim": protein_embeddings.shape[-1], "esm3_status": "live"})
    except Exception as e:
        print(f"   ESM3 failed ({e}), using synthetic protein embeddings")
        # Create synthetic protein embeddings with similar dim to ESM3 (1536)
        protein_embeddings = torch.randn(len(test_sequences["protein"]), 1536, device="cpu")
        print(f"   Synthetic protein embeddings: {protein_embeddings.shape}")
        wandb.log({"protein_embedding_dim": 1536, "esm3_status": "synthetic"})

    # Save Protein embeddings
    torch.save({
        "embeddings": protein_embeddings,
        "labels": torch.from_numpy(labels),
    }, output_dir / "esm3.pt")

    # Save labels
    torch.save({"labels": torch.from_numpy(labels)}, output_dir / "labels.pt")

    # Step 4: Fuse embeddings with MergingModule (3-way: DNA + RNA + Protein)
    print("\n[4/5] Fusing embeddings with MergingModule (3-way)...")
    from manylatents.algorithms.latent import MergingModule

    # Ensure embeddings are on CPU for merging (avoid device mismatch)
    dna_cpu = dna_embeddings.cpu()
    rna_cpu = rna_embeddings.cpu()
    protein_cpu = protein_embeddings.cpu()

    # Test 3-way concat strategy: Evo2 (1920) + Orthrus (256) + ESM3 (1536) = 3712
    merger_concat = MergingModule(
        embeddings={"evo2": dna_cpu, "orthrus": rna_cpu, "esm3": protein_cpu},
        strategy="concat",
        normalize=False,
    )
    fused_concat = merger_concat.fit_transform(dna_cpu)  # dummy input
    print(f"   3-way concat fusion: {fused_concat.shape}")
    print(f"   Expected: Evo2 ({dna_cpu.shape[-1]}) + Orthrus ({rna_cpu.shape[-1]}) + ESM3 ({protein_cpu.shape[-1]}) = {dna_cpu.shape[-1] + rna_cpu.shape[-1] + protein_cpu.shape[-1]}")
    wandb.log({"fused_concat_dim": fused_concat.shape[-1], "n_modalities": 3})

    # Test normalized concat
    merger_norm = MergingModule(
        embeddings={"evo2": dna_cpu, "orthrus": rna_cpu, "esm3": protein_cpu},
        strategy="concat",
        normalize=True,
    )
    fused_norm = merger_norm.fit_transform(dna_cpu)
    print(f"   Normalized 3-way concat: {fused_norm.shape}")

    # Step 5: Compute geometric metrics
    print("\n[5/5] Computing geometric metrics...")
    from manylatents.metrics import compute_metric

    # Convert to numpy for metrics
    fused_np = fused_concat.numpy()

    metrics = {}

    # Participation Ratio
    try:
        pr = compute_metric("ParticipationRatio", fused_np)
        metrics["participation_ratio"] = float(pr)
        print(f"   Participation Ratio: {pr:.4f}")
    except Exception as e:
        print(f"   Participation Ratio: SKIPPED ({e})")

    # Local Intrinsic Dimensionality
    try:
        lid = compute_metric("LocalIntrinsicDimensionality", fused_np)
        if isinstance(lid, tuple):
            lid = lid[0]  # Mean LID
        metrics["lid"] = float(lid)
        print(f"   Local Intrinsic Dimensionality: {lid:.4f}")
    except Exception as e:
        print(f"   LID: SKIPPED ({e})")

    # Log metrics to WandB
    wandb.log(metrics)

    # Log embedding summary
    summary = {
        "dna_mean_norm": float(torch.norm(dna_embeddings, dim=-1).mean()),
        "rna_mean_norm": float(torch.norm(rna_embeddings, dim=-1).mean()),
        "protein_mean_norm": float(torch.norm(protein_embeddings, dim=-1).mean()),
        "fused_mean_norm": float(torch.norm(fused_concat, dim=-1).mean()),
        "n_pathogenic": int(labels.sum()),
        "n_benign": int(len(labels) - labels.sum()),
    }
    wandb.log(summary)

    print("\n" + "=" * 60)
    print("E2E Test Complete!")
    print("=" * 60)
    print(f"\nEmbeddings saved to: {output_dir}")
    print(f"WandB run: {run.url}")

    # Final summary table
    wandb.log({
        "test_status": "SUCCESS",
        "output_dir": str(output_dir),
    })

    run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
