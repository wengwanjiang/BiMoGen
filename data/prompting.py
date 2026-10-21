import random
import torch

import models.configuration as mcfg

def frame_tokens(codes, T_prime):
    n = T_prime * mcfg.FRAME_SIZE
    return [mcfg.MOTION_CODE_START + codes[i] for i in range(n)]

def _build_pretrain(text_tokens, motion_codes, T_prime, t):
    motion = frame_tokens(motion_codes, T_prime)
    text_list = list(text_tokens)

    if random.random() < 0.5:
        seq = [mcfg.CLS_ID, mcfg.SOT_ID] + text_list + [mcfg.EOT_ID, mcfg.SEP_ID] + motion
        maskable = torch.zeros(len(seq), dtype=torch.bool)
        maskable[1] = True                                    # SOT
        maskable[2:2 + len(text_list)] = True                 # text
        maskable[2 + len(text_list)] = True                   # EOT
        mot_start = 4 + len(text_list)
        maskable[mot_start:] = True                            # motion
        ttype = torch.zeros(len(seq), dtype=torch.long)
        ttype[mot_start:] = 1
    else:
        seq = [mcfg.CLS_ID] + motion + [mcfg.SEP_ID, mcfg.SOT_ID] + text_list + [mcfg.EOT_ID]
        maskable = torch.zeros(len(seq), dtype=torch.bool)
        maskable[1:1 + len(motion)] = True                    # motion
        txt_start = 1 + len(motion) + 1
        maskable[txt_start] = True                             # SOT
        maskable[txt_start + 1:txt_start + 1 + len(text_list)] = True
        maskable[txt_start + 1 + len(text_list)] = True        # EOT
        ttype = torch.zeros(len(seq), dtype=torch.long)
        ttype[1:1 + len(motion)] = 1

    L = len(seq)
    input_ids = torch.tensor(seq, dtype=torch.long)
    target_ids = input_ids.clone()

    loss_mask = maskable & (torch.rand(L) < t)
    input_ids[loss_mask] = mcfg.MASK_ID

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": torch.ones(L, dtype=torch.long),
        "loss_mask": loss_mask,
        "token_type_ids": ttype,
        "sc_target_mask": maskable,
    }

def collate_pretrain(batch, min_t=0.0):
    samples = []
    t_values = []
    for item in batch:
        t = random.random() * (1.0 - min_t) + min_t
        s = _build_pretrain(item["text_tokens"], item["motion_codes"],
                            item["T_prime"], t)
        samples.append(s)
        t_values.append(t)

    out = _pad_batch(samples)
    out["mask_ratio_t"] = torch.tensor(t_values, dtype=torch.float32)
    return out

def _build_sft_t2m(text_tokens, motion_codes, T_prime, t, cfg_drop):
    motion = frame_tokens(motion_codes, T_prime)
    text_list = list(text_tokens)
    cond = [mcfg.CLS_ID, mcfg.T2M_ID, mcfg.SOT_ID] + text_list + [mcfg.EOT_ID, mcfg.SEP_ID]
    seq = cond + motion
    L = len(seq)
    vis = len(cond)

    input_ids = torch.tensor(seq, dtype=torch.long)
    target_ids = input_ids.clone()
    ttype = torch.zeros(L, dtype=torch.long)
    ttype[vis:] = 1

    loss_mask = torch.zeros(L, dtype=torch.bool)
    loss_mask[vis:] = torch.rand(len(motion)) < t

    sc_target_mask = torch.zeros(L, dtype=torch.bool)
    sc_target_mask[vis:] = True

    if cfg_drop:
        input_ids[2:4 + len(text_list)] = mcfg.MASK_ID
    input_ids[loss_mask] = mcfg.MASK_ID

    return {"input_ids": input_ids, "target_ids": target_ids,
            "attention_mask": torch.ones(L, dtype=torch.long),
            "loss_mask": loss_mask, "token_type_ids": ttype,
            "sc_target_mask": sc_target_mask}

def _build_sft_m2t(text_tokens, motion_codes, T_prime, t, cfg_drop):
    motion = frame_tokens(motion_codes, T_prime)
    text_list = list(text_tokens)
    cond = [mcfg.CLS_ID, mcfg.M2T_ID] + motion + [mcfg.SEP_ID]
    target = [mcfg.SOT_ID] + text_list + [mcfg.EOT_ID]

    seq = cond + target
    L = len(seq)
    vis = len(cond)

    input_ids = torch.tensor(seq, dtype=torch.long)
    target_ids = input_ids.clone()
    ttype = torch.zeros(L, dtype=torch.long)
    ttype[2:2 + len(motion)] = 1

    loss_mask = torch.zeros(L, dtype=torch.bool)
    tmask = torch.rand(len(target)) < t
    if not tmask.any():
        tmask[random.randint(0, len(target) - 1)] = True
    loss_mask[vis:] = tmask

    sc_target_mask = torch.zeros(L, dtype=torch.bool)
    sc_target_mask[vis:] = True

    if cfg_drop:
        input_ids[2:2 + len(motion)] = mcfg.MASK_ID
    input_ids[loss_mask] = mcfg.MASK_ID

    return {"input_ids": input_ids, "target_ids": target_ids,
            "attention_mask": torch.ones(L, dtype=torch.long),
            "loss_mask": loss_mask, "token_type_ids": ttype,
            "sc_target_mask": sc_target_mask,
            "is_m2t": True, "t_val": t,
            "m2t_target_len": len(target)}

def _build_sft_m2m(cond_codes, cond_tp, target_codes, target_tp, t, cfg_drop):
    cond = frame_tokens(cond_codes, cond_tp)
    tgt = frame_tokens(target_codes, target_tp)
    prefix = [mcfg.CLS_ID, mcfg.M2M_ID] + cond + [mcfg.SEP_ID]
    seq = prefix + tgt
    L = len(seq)
    vis = len(prefix)

    input_ids = torch.tensor(seq, dtype=torch.long)
    target_ids = input_ids.clone()
    ttype = torch.zeros(L, dtype=torch.long)
    ttype[2:2 + len(cond)] = 1
    ttype[vis:] = 1

    loss_mask = torch.zeros(L, dtype=torch.bool)
    loss_mask[vis:] = torch.rand(len(tgt)) < t

    sc_target_mask = torch.zeros(L, dtype=torch.bool)
    sc_target_mask[vis:] = True

    if cfg_drop:
        input_ids[2:2 + len(cond)] = mcfg.MASK_ID
    input_ids[loss_mask] = mcfg.MASK_ID

    return {"input_ids": input_ids, "target_ids": target_ids,
            "attention_mask": torch.ones(L, dtype=torch.long),
            "loss_mask": loss_mask, "token_type_ids": ttype,
            "sc_target_mask": sc_target_mask}

def collate_sft(batch, cfg_prob=0.1, task_weights=(0.8, 0.1, 0.1),
                mask_schedule="uniform", min_t=0.0,
                inbetween_prob=0.5):
    tasks = ["t2m", "m2t", "m2m"]
    samples = []
    t_values = []
    frame_size = mcfg.FRAME_SIZE

    for i, item in enumerate(batch):
        task = random.choices(tasks, weights=task_weights, k=1)[0]
        t = random.random() * (1.0 - min_t) + min_t
        cfg_drop = random.random() < cfg_prob

        if task == "m2m":
            tp = item["T_prime"]
            if tp < 2 * mcfg.MIN_T_PRIME:
                task = "t2m"
            else:
                use_inbetween = (tp >= 3 * mcfg.MIN_T_PRIME and
                                 random.random() < inbetween_prob)
                if use_inbetween:
                    n_head = random.randint(mcfg.MIN_T_PRIME,
                                            tp - 2 * mcfg.MIN_T_PRIME)
                    n_tail = random.randint(mcfg.MIN_T_PRIME,
                                            tp - n_head - mcfg.MIN_T_PRIME)
                    n_middle = tp - n_head - n_tail
                    head_codes = item["motion_codes"][:n_head * frame_size]
                    tail_codes = item["motion_codes"][
                        (tp - n_tail) * frame_size: tp * frame_size]
                    middle_codes = item["motion_codes"][
                        n_head * frame_size:(n_head + n_middle) * frame_size]
                    cond_codes = head_codes + tail_codes
                    cond_tp = n_head + n_tail
                    tgt_codes = middle_codes
                    tgt_tp = n_middle
                else:
                    split = random.randint(mcfg.MIN_T_PRIME,
                                           tp - mcfg.MIN_T_PRIME)
                    cond_tp, tgt_tp = split, tp - split
                    cond_codes = item["motion_codes"][:split * frame_size]
                    tgt_codes = item["motion_codes"][
                        split * frame_size:tp * frame_size]

        if task == "m2m":
            s = _build_sft_m2m(cond_codes, cond_tp, tgt_codes, tgt_tp, t, cfg_drop)
        elif task == "m2t":
            s = _build_sft_m2t(item["text_tokens"], item["motion_codes"],
                               item["T_prime"], t, cfg_drop)
        else:
            s = _build_sft_t2m(item["text_tokens"], item["motion_codes"],
                               item["T_prime"], t, cfg_drop)
        samples.append(s)
        t_values.append(t)

    out = _pad_batch(samples)
    out["mask_ratio_t"] = torch.tensor(t_values, dtype=torch.float32)
    return out

M2T_TARGET_FIXED_LEN = mcfg.MAX_TEXT_CONTENT + 2

def _pad_batch(samples):
    has_m2t = any(s.get("is_m2t", False) for s in samples)
    max_m2t_target = M2T_TARGET_FIXED_LEN if has_m2t else 0

    for s in samples:
        if not s.get("is_m2t", False):
            continue
        tgt_len = int(s["m2t_target_len"])
        if tgt_len >= max_m2t_target:
            continue
        n_extra = max_m2t_target - tgt_len
        t = float(s.get("t_val", 0.5))
        extra_target = torch.full((n_extra,), mcfg.PAD_ID,
                                  dtype=s["target_ids"].dtype)
        extra_input = extra_target.clone()
        extra_loss = torch.rand(n_extra) < t
        if not extra_loss.any():
            extra_loss[random.randint(0, n_extra - 1)] = True
        extra_input[extra_loss] = mcfg.MASK_ID
        extra_attn = torch.ones(n_extra, dtype=s["attention_mask"].dtype)
        extra_ttype = torch.zeros(n_extra, dtype=s["token_type_ids"].dtype)
        s["input_ids"] = torch.cat([s["input_ids"], extra_input])
        s["target_ids"] = torch.cat([s["target_ids"], extra_target])
        s["attention_mask"] = torch.cat([s["attention_mask"], extra_attn])
        s["loss_mask"] = torch.cat([s["loss_mask"], extra_loss])
        s["token_type_ids"] = torch.cat([s["token_type_ids"], extra_ttype])
        if "sc_target_mask" in s:
            extra_sc = torch.ones(n_extra, dtype=s["sc_target_mask"].dtype)
            s["sc_target_mask"] = torch.cat([s["sc_target_mask"], extra_sc])

    B = len(samples)
    max_len = max(s["input_ids"].shape[0] for s in samples)
    keys = ["input_ids", "attention_mask", "target_ids",
            "loss_mask", "token_type_ids"]
    if "sc_target_mask" in samples[0]:
        keys.append("sc_target_mask")
    out = {k: torch.zeros(B, max_len, dtype=samples[0][k].dtype) for k in keys}
    for i, s in enumerate(samples):
        L = s["input_ids"].shape[0]
        for k in out:
            out[k][i, :L] = s[k]
    return out
