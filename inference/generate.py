"""Motion-text generation with confidence-guided decoding."""
import numpy as np
import torch
import torch.nn.functional as F

import models.configuration as mcfg
from data.prompting import frame_tokens


def _schedule(n_mask, steps):
    base = n_mask // steps
    rem = n_mask % steps
    return [base + (1 if i < rem else 0) for i in range(steps)]


def _cfg_logits(model, ids, attn, ttype, cond_s, cond_e, cfg):
    logits = model.get_logits(ids, attn, token_type_ids=ttype)
    if cfg > 1.0:
        u = ids.clone()
        u[:, cond_s:cond_e] = mcfg.MASK_ID
        lu = model.get_logits(u, attn, token_type_ids=ttype)
        logits = lu + cfg * (logits - lu)
    return logits


def _corrector_loop(model, ids, attn, ttype, logits, tgt_s, tgt_e,
                    corrector_steps):
    """Apply self-correction to the target sequence."""
    y = logits.argmax(dim=-1)
    corr_logits = logits
    for _ in range(corrector_steps):
        corr_logits = model.get_logits(y, attn, token_type_ids=ttype)
        y_new = corr_logits.argmax(dim=-1)
        if (y_new == y).all():
            break
        y = y_new

    for pos in range(tgt_s, tgt_e):
        if ids[0, pos] != mcfg.MASK_ID:
            ids[0, pos] = y[0, pos]

    return ids, corr_logits


def _decode(model, ids, attn, ttype, cond_s, cond_e,
            tgt_s, tgt_e, steps, cfg, logits_proc=None,
            corrector_steps=0, corrector_freq=1, corrector_time=None,
            temperature=0.0):
    """Decode masked tokens over the requested number of steps."""
    n_mask = (ids[0, tgt_s:tgt_e] == mcfg.MASK_ID).sum().item()
    if n_mask == 0:
        return
    schedule = _schedule(n_mask, steps)

    for step, k in enumerate(schedule):
        logits = _cfg_logits(model, ids, attn, ttype, cond_s, cond_e, cfg)
        if logits_proc:
            logits = logits_proc(logits)

        if corrector_steps > 0:
            step1 = step + 1  # 1-indexed
            if corrector_time is not None:
                do_corr = step1 in corrector_time
            else:
                do_corr = step1 % corrector_freq == 0
            if do_corr:
                ids, logits = _corrector_loop(
                    model, ids, attn, ttype, logits, tgt_s, tgt_e,
                    corrector_steps)

        if temperature and temperature > 0:
            scaled = logits.float() / temperature
            sample_prob = F.softmax(scaled, dim=-1)
            B, L, V = sample_prob.shape
            pred = torch.multinomial(
                sample_prob.view(-1, V), num_samples=1
            ).view(B, L)
            prob = F.softmax(logits.float(), dim=-1)
            conf = prob.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        else:
            pred = logits.argmax(dim=-1)
            prob = F.softmax(logits.float(), dim=-1)
            conf = prob.gather(-1, pred.unsqueeze(-1)).squeeze(-1)

        mask = (ids == mcfg.MASK_ID)
        pred = torch.where(mask, pred, ids)
        conf = torch.where(mask, conf, torch.tensor(-np.inf, device=conf.device))

        if k > 0:
            _, sel = torch.topk(conf[0], k=min(k, mask[0].sum().item()))
            ids[0, sel] = pred[0, sel]


M2T_MAX = mcfg.MAX_TEXT_CONTENT


@torch.no_grad()
def generate_t2m(model, text_ids, num_steps=20, cfg_scale=3.0,
                 target_T_prime=None, device="cuda",
                 corrector_steps=0, corrector_freq=1,
                 corrector_time=None, temperature=0.0, **kw):
    """Generate motion tokens conditioned on text."""
    assert target_T_prime is not None
    model.eval()
    text = list(text_ids)
    prefix = [mcfg.CLS_ID, mcfg.T2M_ID, mcfg.SOT_ID] + text + [mcfg.EOT_ID, mcfg.SEP_ID]
    n_mot = target_T_prime * mcfg.FRAME_SIZE
    seq = prefix + [mcfg.MASK_ID] * n_mot

    ids = torch.tensor([seq], dtype=torch.long, device=device)
    L = ids.shape[1]
    attn = torch.ones(1, L, dtype=torch.long, device=device)
    vis = len(prefix)

    ttype = torch.zeros(1, L, dtype=torch.long, device=device)
    ttype[:, vis:] = 1

    _decode(model, ids, attn, ttype,
            2, 4 + len(text),
            vis, vis + n_mot, num_steps, cfg_scale,
            corrector_steps=corrector_steps,
            corrector_freq=corrector_freq,
            corrector_time=corrector_time,
            temperature=temperature)
    return ids[0, vis:vis + n_mot].cpu().tolist()


@torch.no_grad()
def generate_m2t(model, motion_codes, T_prime, num_steps=20, cfg_scale=1.0,
                 device="cuda",
                 corrector_steps=0, corrector_freq=1,
                 corrector_time=None, max_text_len=None,
                 temperature=0.0, **kw):
    """Generate text tokens conditioned on motion."""
    model.eval()
    mot = frame_tokens(motion_codes, T_prime)
    text_slots = M2T_MAX if max_text_len is None else int(max_text_len)
    text_slots = max(1, min(text_slots, M2T_MAX))
    n_mask = 1 + text_slots + 1
    prefix = [mcfg.CLS_ID, mcfg.M2T_ID] + mot + [mcfg.SEP_ID]
    seq = prefix + [mcfg.MASK_ID] * n_mask

    ids = torch.tensor([seq], dtype=torch.long, device=device)
    L = ids.shape[1]
    attn = torch.ones(1, L, dtype=torch.long, device=device)
    vis = len(prefix)

    ttype = torch.zeros(1, L, dtype=torch.long, device=device)
    ttype[:, 2:2 + len(mot)] = 1

    _decode(model, ids, attn, ttype,
            2, 2 + len(mot),
            vis, L, num_steps, cfg_scale,
            corrector_steps=corrector_steps,
            corrector_freq=corrector_freq,
            corrector_time=corrector_time,
            temperature=temperature)

    result = ids[0, vis:].cpu().tolist()
    for i, t in enumerate(result):
        if t == mcfg.EOT_ID:
            result = result[:i]
            break
    if result and result[0] == mcfg.SOT_ID:
        result = result[1:]
    return result[:text_slots]


@torch.no_grad()
def generate_m2m(model, cond_codes, cond_tp, num_steps=20,
                 target_T_prime=None, device="cuda", cfg_scale=1.0,
                 corrector_steps=0, corrector_freq=1,
                 corrector_time=None, **kw):
    """Generate motion tokens conditioned on a motion segment."""
    assert target_T_prime is not None
    model.eval()
    cond = frame_tokens(cond_codes, cond_tp)
    prefix = [mcfg.CLS_ID, mcfg.M2M_ID] + cond + [mcfg.SEP_ID]
    n_mot = target_T_prime * mcfg.FRAME_SIZE
    seq = prefix + [mcfg.MASK_ID] * n_mot

    ids = torch.tensor([seq], dtype=torch.long, device=device)
    L = ids.shape[1]
    attn = torch.ones(1, L, dtype=torch.long, device=device)
    vis = len(prefix)

    ttype = torch.zeros(1, L, dtype=torch.long, device=device)
    ttype[:, 2:2 + len(cond)] = 1
    ttype[:, vis:vis + n_mot] = 1

    _decode(model, ids, attn, ttype,
            2, 2 + len(cond),
            vis, vis + n_mot, num_steps, cfg_scale,
            corrector_steps=corrector_steps,
            corrector_freq=corrector_freq,
            corrector_time=corrector_time)
    return ids[0, vis:vis + n_mot].cpu().tolist()


@torch.no_grad()
def generate_pred(model, motion_codes, T_prime, predict_ratio=0.2,
                  num_steps=20, device="cuda", cfg_scale=1.0,
                  corrector_steps=0, corrector_freq=1,
                  corrector_time=None, **kw):
    """Complete motion after a visible prefix."""
    model.eval()
    all_codes = frame_tokens(motion_codes, T_prime)
    n_cond = max(1, int(len(all_codes) * predict_ratio))
    cond = all_codes[:n_cond]
    n_target = len(all_codes) - n_cond

    prefix = [mcfg.CLS_ID, mcfg.M2M_ID] + cond + [mcfg.SEP_ID]
    seq = prefix + [mcfg.MASK_ID] * n_target

    ids = torch.tensor([seq], dtype=torch.long, device=device)
    L = ids.shape[1]
    attn = torch.ones(1, L, dtype=torch.long, device=device)
    vis = len(prefix)

    ttype = torch.zeros(1, L, dtype=torch.long, device=device)
    ttype[:, 2:2 + len(cond)] = 1
    ttype[:, vis:vis + n_target] = 1

    _decode(model, ids, attn, ttype,
            2, 2 + len(cond),
            vis, vis + n_target, num_steps, cfg_scale,
            corrector_steps=corrector_steps,
            corrector_freq=corrector_freq,
            corrector_time=corrector_time)

    gen_tokens = ids[0, vis:vis + n_target].cpu().tolist()
    return all_codes[:n_cond] + gen_tokens


@torch.no_grad()
def generate_inbetween(model, motion_codes, T_prime, inbetween_ratio=0.25,
                       num_steps=20, device="cuda", cfg_scale=1.0,
                       corrector_steps=0, corrector_freq=1,
                       corrector_time=None, **kw):
    """Complete motion between visible endpoint segments."""
    model.eval()
    all_codes = frame_tokens(motion_codes, T_prime)
    n_total = len(all_codes)
    n_head = max(1, int(n_total * inbetween_ratio))
    n_tail = max(1, int(n_total * inbetween_ratio))
    n_middle = n_total - n_head - n_tail
    if n_middle <= 0:
        return all_codes

    head = all_codes[:n_head]
    tail = all_codes[n_total - n_tail:]
    cond = head + tail

    prefix = [mcfg.CLS_ID, mcfg.M2M_ID] + cond + [mcfg.SEP_ID]
    seq = prefix + [mcfg.MASK_ID] * n_middle

    ids = torch.tensor([seq], dtype=torch.long, device=device)
    L = ids.shape[1]
    attn = torch.ones(1, L, dtype=torch.long, device=device)
    vis = len(prefix)

    ttype = torch.zeros(1, L, dtype=torch.long, device=device)
    ttype[:, 2:2 + len(cond)] = 1
    ttype[:, vis:vis + n_middle] = 1

    _decode(model, ids, attn, ttype,
            2, 2 + len(cond),
            vis, vis + n_middle, num_steps, cfg_scale,
            corrector_steps=corrector_steps,
            corrector_freq=corrector_freq,
            corrector_time=corrector_time)

    gen_middle = ids[0, vis:vis + n_middle].cpu().tolist()
    return head + gen_middle + tail
