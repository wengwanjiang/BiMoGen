"""Shared model construction and checkpoint loading for all entry points."""
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import models.configuration as cfg
from models.motion_llada import MotionLLaDA


def build_model(config):
    m = config['model']
    return MotionLLaDA(
        d_model=m.get('llada_d_model', 1024), n_heads=m.get('llada_n_heads', 16),
        n_layers=m.get('llada_n_layers', 20),
        mlp_hidden_size=m.get('llada_mlp_hidden_size', 3072),
        dropout=m.get('dropout', 0.0),
        pad_loss_weight=config.get('task', {}).get('pad_loss_weight', 0.1),
    )


def load_tokenizer(path):
    path = Path(path)
    if (path / 'tokenizer.json').is_file():
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
    cfg.setup_tokenizer(tokenizer)
    return tokenizer


def load_model(checkpoint_path, config_path=None, device='cpu', vq_override=None):
    if vq_override not in (None, 'frame'):
        raise ValueError('Only frame VQ is supported.')
    path = Path(checkpoint_path)
    state = torch.load(path / 'state.pt', map_location='cpu', weights_only=False)
    config = state.get('config')
    if config is None:
        with open(config_path or path / 'config.yaml') as f:
            config = yaml.safe_load(f)
    if config_path is not None:
        # Override runtime assets, preserving the checkpoint architecture.
        with open(config_path) as f:
            runtime = yaml.safe_load(f)
        for section in ('data', 'vqvae'):
            config[section] = {**config.get(section, {}), **runtime.get(section, {})}
    if config.get('vq_mode', 'frame') != 'frame':
        raise ValueError('BiMoGen uses the single-layer frame VQ-VAE.')
    tokenizer = load_tokenizer(path / 'tokenizer')
    model = build_model(config)
    model.load_state_dict(state['model'], strict=True)
    if 'ema' in state:
        shadow = state['ema']['shadow']
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(shadow[name])
    model.to(device).eval()
    return model, config, state.get('step', 0), tokenizer, 'llada'
