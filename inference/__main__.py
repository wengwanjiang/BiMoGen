"""Generate motion features or captions from a BiMoGen checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch

import models.configuration as cfg
from inference.generate import generate_t2m, generate_m2t
from models.loading import load_model
from models.motion_vqvae import load_motion_vqvae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text', help='Text prompt for motion generation.')
    inputs.add_argument('--motion', help='Raw, unnormalized motion features (.npy).')
    parser.add_argument('--frames', type=int, default=196)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='output/motion.npy')
    args = parser.parse_args()
    model, config, _, tokenizer, _ = load_model(args.checkpoint, device=args.device)
    dc = config['data']
    vq = load_motion_vqvae(config['vqvae']['checkpoint'], nfeats=dc['nfeats'], device=args.device)
    mean, std = np.load(dc['mean_path']), np.load(dc['std_path'])
    with torch.inference_mode():
        if args.text is not None:
            if args.frames < 4 or args.frames > cfg.MAX_MOTION_FRAMES or args.frames % 4:
                parser.error('--frames must be a multiple of 4 between 4 and 196.')
            text = tokenizer(args.text, add_special_tokens=False, truncation=True,
                             max_length=cfg.MAX_TEXT_CONTENT)['input_ids']
            tokens = generate_t2m(model, text, target_T_prime=args.frames // 4,
                                  num_steps=args.steps, cfg_scale=4.0, device=args.device)
            codes = torch.tensor(tokens, device=args.device) - cfg.MOTION_CODE_START
            motion = vq.decode(codes.clamp(0, cfg.CODEBOOK_SIZE - 1))[0].cpu().numpy() * std + mean
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, motion)
            print(path)
        else:
            motion = np.load(args.motion)
            if motion.ndim != 2 or motion.shape[1] != dc['nfeats']:
                parser.error(f"Motion must have shape (frames, {dc['nfeats']}).")
            length = min(len(motion), cfg.MAX_MOTION_FRAMES) // 4 * 4
            if length < 4:
                parser.error('Motion must contain at least four frames.')
            motion = motion[:length]
            features = torch.from_numpy((motion - mean) / std).float()[None].to(args.device)
            codes, _ = vq.encode(features)
            tokens = generate_m2t(model, codes[0].tolist(), codes.shape[1],
                                  num_steps=args.steps, device=args.device)
            print(tokenizer.decode(tokens, skip_special_tokens=True))


if __name__ == '__main__':
    main()
