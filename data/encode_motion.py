"""Cache HumanML3D motion tokens with the frozen MotionGPT VQ-VAE."""
import argparse
from pathlib import Path

import yaml

from data.dataset import MotionTextDataset
from models.motion_vqvae import load_motion_vqvae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/bimogen.yaml')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                        choices=['train', 'val', 'test'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    dc = config['data']
    root = Path(dc['data_root'])
    pending = []
    for split in dict.fromkeys(args.splits):
        cache = root / f'motiongpt_vq_codes_{split}.pt'
        if cache.is_file():
            print(f'Using existing cache: {cache}. Remove it to re-encode this split.')
            continue
        if not (root / f'{split}.txt').is_file():
            parser.error(f'Missing split file: {root / (split + ".txt")}')
        pending.append((split, cache))
    if not pending:
        return
    for directory in ('new_joint_vecs', 'texts'):
        if not (root / directory).is_dir():
            parser.error(f'Missing HumanML3D directory: {root / directory}')
    vqvae = load_motion_vqvae(config['vqvae']['checkpoint'],
                            nfeats=dc['nfeats'], device=args.device)
    for split, cache in pending:
        # Reuse the training dataset's normalization, cropping, and cache format.
        # Text tokenization only happens in __getitem__, so no text tokenizer is needed here.
        dataset = MotionTextDataset(
            data_root=str(root), split=split, tokenizer=None, vqvae=vqvae,
            mean_path=dc['mean_path'], std_path=dc['std_path'], device=args.device,
        )
        print(f'Saved {cache}: {len(dataset.codes)} encoded motions; '
              f'{len(dataset)} paired samples.')


if __name__ == '__main__':
    main()
