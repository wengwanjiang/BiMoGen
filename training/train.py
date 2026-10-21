"""Two-stage BiMoGen training. Run with python -m training.train."""
import argparse
import logging
import math
from functools import partial
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from data.dataset import MotionTextDataset
from data.prompting import collate_pretrain, collate_sft
from models.loading import build_model, load_tokenizer
from models.motion_vqvae import load_motion_vqvae

log = logging.getLogger(__name__)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].lerp_(param.detach(), 1 - self.decay)

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}


def save_checkpoint(path, accelerator, model, optimizer, scheduler, ema,
                    config, tokenizer, stage, step, epoch, batch_index):
    accelerator.wait_for_everyone()
    path.mkdir(parents=True, exist_ok=True)
    # Accelerate saves per-rank RNG, optimizer/scaler, and prepared model states.
    accelerator.save_state(str(path / 'training_state'))
    if accelerator.is_main_process:
        state = dict(model=accelerator.unwrap_model(model).state_dict(), config=config,
                     stage=stage, step=step, epoch=epoch, batch_index=batch_index,
                     scheduler=scheduler.state_dict())
        if ema:
            state['ema'] = ema.state_dict()
        torch.save(state, path / 'state.pt.tmp')
        (path / 'state.pt.tmp').replace(path / 'state.pt')
        (path / 'config.yaml').write_text(yaml.safe_dump(config))
        tokenizer.save_pretrained(path / 'tokenizer')
    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/bimogen.yaml')
    parser.add_argument('--stage', choices=['pretrain', 'sft'], required=True)
    parser.add_argument('--resume', help='Resume this stage, or initialize SFT from pretraining.')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    tc, dc, stage = config['training'], config['data'], config[args.stage]
    if args.output_dir:
        tc['output_dir'] = args.output_dir
    acc = Accelerator(mixed_precision=tc['mixed_precision'],
                      gradient_accumulation_steps=tc.get('gradient_accumulation_steps', 1),
                      dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True,
                                                               data_seed=tc['seed']))
    logging.basicConfig(level=logging.INFO if acc.is_main_process else logging.WARNING)
    set_seed(tc['seed'])
    tokenizer = load_tokenizer(Path(args.resume) / 'tokenizer' if args.resume else config['model']['tokenizer'])
    model = build_model(config)
    saved = None
    same_stage = False
    if args.resume:
        saved = torch.load(Path(args.resume) / 'state.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        same_stage = saved.get('stage') == args.stage and (Path(args.resume) / 'training_state').is_dir()
    elif args.stage == 'sft':
        raise ValueError('SFT requires --resume pointing to a pretraining checkpoint.')
    # Build the cache once. Other ranks read it after the main process completes.
    with acc.main_process_first():
        vq = load_motion_vqvae(config['vqvae']['checkpoint'], nfeats=dc['nfeats'])
        dataset = MotionTextDataset(
            data_root=dc['data_root'], split='train', tokenizer=tokenizer, vqvae=vq,
            mean_path=dc['mean_path'], std_path=dc['std_path'], device=acc.device,
            max_text_len=dc.get('max_text_len', 50),
        )
        del vq
    collate = partial(collate_pretrain, min_t=config['gasc']['min_t']) if args.stage == 'pretrain' else partial(
        collate_sft, cfg_prob=config['task']['cfg_prob'],
        task_weights=config['task']['weights'], min_t=config['gasc']['min_t'],
        inbetween_prob=config['task']['m2m_inbetween_prob'])
    loader = DataLoader(dataset, batch_size=tc['batch_size'], shuffle=True,
                        collate_fn=collate, num_workers=dc['num_workers'], drop_last=True,
                        pin_memory=acc.device.type == 'cuda',
                        generator=torch.Generator().manual_seed(tc['seed']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage['lr'],
                                  weight_decay=tc['weight_decay'])
    model, optimizer, loader = acc.prepare(model, optimizer, loader)
    if not len(loader):
        raise ValueError('Dataset is smaller than a training batch.')
    updates_per_epoch = math.ceil(len(loader) / acc.gradient_accumulation_steps)
    total = updates_per_epoch * stage['epochs']
    warmup = min(stage['warmup_steps'], max(total - 1, 0))

    def lr_scale(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = min((step - warmup) / max(total - warmup, 1), 1)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lr_scale)
    raw = acc.unwrap_model(model)
    ema = EMA(raw, config['ema']['decay']) if config['ema']['enabled'] else None
    start_epoch = start_batch = step = 0
    if same_stage:
        state_dir = Path(args.resume) / 'training_state'
        if not state_dir.is_dir():
            raise ValueError('Exact resume requires a checkpoint with training_state/.')
        acc.load_state(str(state_dir))
        scheduler.load_state_dict(saved['scheduler'])
        start_epoch, start_batch, step = saved['epoch'], saved['batch_index'], saved['step']
        if ema:
            ema.decay = saved['ema']['decay']
            ema.shadow = {n: p.to(acc.device) for n, p in saved['ema']['shadow'].items()}
    del saved
    out = Path(tc['output_dir'])
    model.train()
    for epoch in range(start_epoch, stage['epochs']):
        loader.set_epoch(epoch)
        offset = start_batch if epoch == start_epoch else 0
        batches = acc.skip_first_batches(loader, offset) if offset else loader
        for batch_index, batch in enumerate(batches, start=offset):
            with acc.accumulate(model):
                loss = model(**batch, sc_loss_weight=config['gasc']['weight'],
                             use_t_weighting=config['gasc']['use_t_weighting'])
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), tc['gradient_clip'])
                optimizer.step()
                optimizer.zero_grad()
            if not acc.sync_gradients:
                continue
            if acc.optimizer_step_was_skipped:
                continue
            scheduler.step()
            step += 1
            if ema:
                ema.update(raw)
            if step % tc['log_every'] == 0:
                value = acc.reduce(loss.detach(), reduction='mean').item()
                log.info('%s epoch=%d step=%d/%d loss=%.4f', args.stage, epoch + 1, step, total, value)
            next_epoch, next_batch = epoch, batch_index + 1
            if next_batch == len(loader):
                next_epoch, next_batch = epoch + 1, 0
            if step % tc['save_every'] == 0:
                save_checkpoint(out / f'{args.stage}-checkpoint-{step}', acc, model,
                                optimizer, scheduler, ema, config, tokenizer, args.stage,
                                step, next_epoch, next_batch)
    save_checkpoint(out / f'{args.stage}-final', acc, model, optimizer, scheduler, ema,
                    config, tokenizer, args.stage, step, stage['epochs'], 0)
    acc.end_training()


if __name__ == '__main__':
    main()
