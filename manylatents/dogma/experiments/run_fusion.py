"""Run Central Dogma Fusion experiment.

Encodes sequences using all 3 central dogma foundation models (Evo2, Orthrus, ESM3)
and concatenates the embeddings.

Usage:
    # Default (GFP sequence)
    python -m manylatents.dogma.experiments.run_fusion

    # With specific preset
    python -m manylatents.dogma.experiments.run_fusion --preset synthetic_8aa
    python -m manylatents.dogma.experiments.run_fusion --preset brca1

    # With normalization
    python -m manylatents.dogma.experiments.run_fusion --normalize

    # Save embeddings
    python -m manylatents.dogma.experiments.run_fusion --output embeddings.pt

GPU Requirements:
    - ~24GB+ VRAM to load all 3 encoders simultaneously
    - Recommend A100/L40S/H100 GPUs
"""

import argparse
import logging
import os
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run Central Dogma Fusion experiment')
    parser.add_argument('--preset', type=str, default='gfp',
                        choices=['synthetic_8aa', 'gfp', 'brca1'],
                        help='Preset sequence to encode')
    parser.add_argument('--normalize', action='store_true',
                        help='L2-normalize each modality before concatenation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for embeddings (.pt format)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    args = parser.parse_args()

    logger.info(f'Running Central Dogma Fusion with preset: {args.preset}')

    # Import here to avoid loading models at module import time
    from manylatents.dogma.algorithms import CentralDogmaFusion
    from manylatents.dogma.data import CentralDogmaDataModule

    # Create datamodule
    logger.info('Creating CentralDogmaDataModule...')
    dm = CentralDogmaDataModule(preset=args.preset)
    dm.setup()

    sequences = dm.get_sequences()
    logger.info(f'Sequences loaded:')
    logger.info(f'  DNA length: {len(sequences["dna"])} bp')
    logger.info(f'  RNA length: {len(sequences["rna"])} nt')
    logger.info(f'  Protein length: {len(sequences["protein"])} aa')

    # Create fusion algorithm
    logger.info('Creating CentralDogmaFusion...')
    fusion = CentralDogmaFusion(
        evo2_config={
            '_target_': 'manylatents.dogma.encoders.Evo2Encoder',
            'model_name': 'evo2_1b_base',
            'device': args.device,
        },
        orthrus_config={
            '_target_': 'manylatents.dogma.encoders.OrthrusEncoder',
            'n_tracks': 4,
            'device': args.device,
        },
        esm3_config={
            '_target_': 'manylatents.dogma.encoders.ESM3Encoder',
            'device': args.device,
        },
        normalize=args.normalize,
        datamodule=dm,
    )

    # Fit (no-op for pretrained encoders)
    dummy_tensor = torch.zeros(1, 1)
    fusion.fit(dummy_tensor)

    # Transform (encode and concatenate)
    logger.info('Encoding sequences...')
    logger.info('  Loading Evo2 (DNA)...')
    logger.info('  Loading Orthrus (RNA)...')
    logger.info('  Loading ESM3 (Protein)...')

    embeddings = fusion.transform(dummy_tensor)

    logger.info(f'Fusion complete!')
    logger.info(f'  Embedding shape: {embeddings.shape}')
    logger.info(f'  Expected: (1, 3840) = DNA(2048) + RNA(256) + Protein(1536)')

    # Get individual embeddings for analysis
    individual = fusion.get_embeddings(dummy_tensor)
    logger.info(f'Individual embedding shapes:')
    logger.info(f'  DNA (Evo2): {individual.dna.shape}')
    logger.info(f'  RNA (Orthrus): {individual.rna.shape}')
    logger.info(f'  Protein (ESM3): {individual.protein.shape}')

    # Save embeddings if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'fused_embeddings': embeddings,
            'dna_embeddings': individual.dna,
            'rna_embeddings': individual.rna,
            'protein_embeddings': individual.protein,
            'preset': args.preset,
            'sequences': sequences,
            'normalize': args.normalize,
        }
        torch.save(save_dict, output_path)
        logger.info(f'Embeddings saved to: {output_path}')

    return embeddings


if __name__ == '__main__':
    main()
