"""M2T evaluation: caption quality, retrieval, and matching distance."""
import argparse
import os
import sys
import random
import logging
import json
import math
from collections import Counter
import pickle
import subprocess
import tempfile
import textwrap
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import yaml

import models.configuration as mcfg
from inference.generate import generate_m2t
from evaluation.eval_t2m import load_model
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from utils.get_eval_option import get_opt
from evaluation.metrics.utils import (
    euclidean_distance_matrix, calculate_top_k,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_m2t")

DEFAULT_EVAL_OPT = "deps/t2m/t2m/Comp_v6_KLD005/opt.txt"
DEFAULT_EVAL_CKPT_DIR = "deps/t2m"
DEFAULT_GLOVE_DIR = "deps/glove"
DEFAULT_EVAL_META = "deps/t2m/t2m/Comp_v6_KLD005/meta"

MAX_MOTION_LENGTH = 196
UNIT_LENGTH = 4

_SPACY_NLP = None


def _get_spacy():
    """Load the English spaCy pipeline for caption tokenization."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        try:
            _SPACY_NLP = spacy.load('en_core_web_sm')
        except OSError as e:
            raise OSError(
                "spaCy model 'en_core_web_sm' is not installed in the current "
                "Python environment. Install it with:\n"
                "    python -m spacy download en_core_web_sm\n"
                f"Original error: {e}"
            ) from e
    return _SPACY_NLP


def compute_bleu(references_list, hypotheses, max_n=4):
    """Compute corpus BLEU-1 through BLEU-N with pycocoevalcap."""
    try:
        from pycocoevalcap.bleu.bleu import Bleu
    except ImportError:
        logger.warning(
            "pycocoevalcap is not installed; falling back to NaN BLEU. "
            "Install with: pip install pycocoevalcap")
        return {f"Bleu_{n}": float("nan") for n in range(1, max_n + 1)}

    gts = {i: list(refs) for i, refs in enumerate(references_list)}
    res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
    bleu_scores, _ = Bleu(max_n).compute_score(gts, res)
    return {f"Bleu_{n}": float(bleu_scores[n - 1]) for n in range(1, max_n + 1)}


def compute_rouge_l(references_list, hypotheses):
    """Compute ROUGE-L with pycocoevalcap."""
    try:
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError:
        logger.warning(
            "pycocoevalcap is not installed; falling back to NaN ROUGE-L. "
            "Install with: pip install pycocoevalcap")
        return {"Rouge_L": float("nan")}

    gts = {i: list(refs) for i, refs in enumerate(references_list)}
    res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
    score, _ = Rouge().compute_score(gts, res)
    return {"Rouge_L": float(score)}


def _tokenize(text):
    return text.lower().strip().split()

def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def compute_cider(references_list, hypotheses, max_n=4):
    """Compute corpus-level caption consensus scores."""
    num_docs = len(references_list)
    doc_freq = {n: Counter() for n in range(1, max_n + 1)}
    for refs in references_list:
        for n in range(1, max_n + 1):
            seen = set()
            for ref in refs:
                for ng in _ngrams(_tokenize(ref), n):
                    seen.add(ng)
            for ng in seen:
                doc_freq[n][ng] += 1

    def _tfidf(text, n):
        toks = _tokenize(text)
        ngs = _ngrams(toks, n)
        tf = Counter(ngs)
        total = max(len(ngs), 1)
        vec = {}
        for ng, count in tf.items():
            df = doc_freq[n].get(ng, 0)
            if df > 0:
                vec[ng] = (count / total) * math.log(num_docs / df)
        return vec

    def _cosine(v1, v2):
        common = set(v1) & set(v2)
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        n1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        n2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

    scores = []
    for refs, hyp in zip(references_list, hypotheses):
        total = 0.0
        for n in range(1, max_n + 1):
            hv = _tfidf(hyp, n)
            sims = [_cosine(hv, _tfidf(r, n)) for r in refs]
            total += np.mean(sims) / max_n
        scores.append(total)
    return {"CIDEr": float(np.mean(scores) * 10)}


_BERT_SCORE_WORKER = textwrap.dedent("""
    import os, sys, pickle
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, 'rb') as f:
        payload = pickle.load(f)
    from bert_score import score as bert_score_fn
    _, _, F1 = bert_score_fn(
        payload['hyps'], payload['refs'], lang='en',
        rescale_with_baseline=True, idf=True,
        device=payload['device'], verbose=False,
        batch_size=payload['batch_size'])
    with open(out_path, 'wb') as f:
        pickle.dump({'F1': F1.detach().cpu().tolist()}, f)
""").strip()


def _bert_score_subprocess(hyps, refs, device, batch_size):
    """Compute BERTScore in an isolated worker process."""
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, 'in.pkl')
        out_path = os.path.join(td, 'out.pkl')
        with open(in_path, 'wb') as f:
            pickle.dump({'hyps': hyps, 'refs': refs,
                         'device': device, 'batch_size': batch_size}, f)
        cmd = [sys.executable, '-c', _BERT_SCORE_WORKER, in_path, out_path]
        env = os.environ.copy()
        env.setdefault('PYTHONUNBUFFERED', '1')
        proc = subprocess.run(cmd, env=env, capture_output=True)
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(
                f"bert-score worker exited with code {proc.returncode}.\n"
                f"stderr tail:\n{proc.stderr.decode(errors='replace')[-1000:]}"
            )
        with open(out_path, 'rb') as f:
            return pickle.load(f)['F1']


def compute_bert_score(references_list, hypotheses, device="cuda",
                       batch_size=64, isolate=True):
    """Compute single-reference BERTScore F1 with baseline rescaling and IDF."""
    if not hypotheses:
        return {"BERTScore": float("nan")}

    if isinstance(references_list[0], (list, tuple)):
        refs_single = [r[0] if r else "" for r in references_list]
    else:
        refs_single = list(references_list)

    if isolate:
        try:
            f1 = _bert_score_subprocess(hypotheses, refs_single,
                                         str(device), batch_size)
        except RuntimeError as e:
            logger.warning(f"BERTScore failed in subprocess, returning NaN. "
                           f"Reason: {e}")
            return {"BERTScore": float("nan")}
        if not f1:
            return {"BERTScore": float("nan")}
        return {"BERTScore": float(np.mean(f1))}

    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.warning("bert-score not installed, skipping BERTScore.")
        return {"BERTScore": float("nan")}

    _, _, F1 = bert_score_fn(hypotheses, refs_single, lang="en",
                             rescale_with_baseline=True, idf=True,
                             device=str(device), verbose=False,
                             batch_size=batch_size)
    return {"BERTScore": float(F1.mean().item())}


def caption_to_evaluator_input(caption, w_vectorizer, max_text_len=20):
    """Convert a caption to lemmatized GloVe embeddings and POS features."""
    sentence = caption.replace('-', '')
    nlp = _get_spacy()
    doc = nlp(sentence)

    word_list, pos_list = [], []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and word != 'left':
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)

    t_tokens = [f"{w}/{p}" for w, p in zip(word_list, pos_list)]

    if len(t_tokens) < max_text_len:
        tokens = ["sos/OTHER"] + t_tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens += ["unk/OTHER"] * (max_text_len + 2 - sent_len)
    else:
        tokens = t_tokens[:max_text_len]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)

    we_list, po_list = [], []
    for tok in tokens:
        try:
            wv, pv = w_vectorizer[tok]
        except Exception:
            wv = np.zeros(300, dtype=np.float32)
            pv = np.zeros(15, dtype=np.float32)
        we_list.append(wv)
        po_list.append(pv)

    return (torch.from_numpy(np.stack(we_list).astype(np.float32)),
            torch.from_numpy(np.stack(po_list).astype(np.float32)),
            sent_len)


def compute_alignment_metrics(gen_text_embs, gt_text_embs, motion_embs,
                              top_k=3, R_size=32):
    """R-Precision and MM-Dist for M2T (generated text vs GT motion)."""
    N = len(gen_text_embs)
    idx = np.random.permutation(N)
    gen_text = gen_text_embs[idx]
    gt_text = gt_text_embs[idx]
    motions = motion_embs[idx]

    R_count = (N // R_size) * R_size
    gen_match, gt_match = 0.0, 0.0
    gen_topk = np.zeros(top_k)
    gt_topk = np.zeros(top_k)

    for i in range(N // R_size):
        s, e = i * R_size, (i + 1) * R_size
        gt = torch.from_numpy(gen_text[s:e])
        gtt = torch.from_numpy(gt_text[s:e])
        m = torch.from_numpy(motions[s:e])

        d = euclidean_distance_matrix(gt, m).nan_to_num()
        gen_match += d.trace().item()
        gen_topk += calculate_top_k(
            torch.argsort(d, dim=1), top_k=top_k).sum(axis=0).numpy()

        d2 = euclidean_distance_matrix(gtt, m).nan_to_num()
        gt_match += d2.trace().item()
        gt_topk += calculate_top_k(
            torch.argsort(d2, dim=1), top_k=top_k).sum(axis=0).numpy()

    gen_r = gen_topk / R_count
    gt_r = gt_topk / R_count
    return {
        "R1": float(gen_r[0]), "R2": float(gen_r[1]), "R3": float(gen_r[2]),
        "MM-Dist": float(gen_match / R_count),
        "gt_R1": float(gt_r[0]), "gt_R2": float(gt_r[1]),
        "gt_R3": float(gt_r[2]),
        "gt_MM-Dist": float(gt_match / R_count),
    }


def ensure_vq_cache(data_root, split, cfg, device):
    """Return path to VQ-code cache, encoding test set if needed."""
    prefix = "motiongpt"
    cache = os.path.join(data_root, f"{prefix}_vq_codes_{split}.pt")
    if os.path.exists(cache):
        return cache

    logger.info(f"VQ cache not found, encoding {split} set …")
    vq_ckpt = cfg["vqvae"]["checkpoint"]
    from models.motion_vqvae import load_motion_vqvae
    vqvae = load_motion_vqvae(vq_ckpt, device=str(device))
    vqvae.eval()

    mean = np.load(cfg["data"]["mean_path"])
    std = np.load(cfg["data"]["std_path"])
    with open(os.path.join(data_root, f"{split}.txt")) as f:
        ids = [l.strip() for l in f if l.strip()]

    codes_d, tp_d = {}, {}
    for mid in tqdm(ids, desc="Encoding"):
        npy = os.path.join(data_root, "new_joint_vecs", f"{mid}.npy")
        if not os.path.exists(npy):
            continue
        mot = np.load(npy)
        T = min(mot.shape[0], mcfg.MAX_MOTION_FRAMES)
        if T < mcfg.MIN_T_PRIME * mcfg.TEMPORAL_DOWNSAMPLE:
            continue
        mot = mot[:T // mcfg.TEMPORAL_DOWNSAMPLE * mcfg.TEMPORAL_DOWNSAMPLE]
        normed = (mot - mean) / std
        feat = torch.from_numpy(normed).float().unsqueeze(0).to(device)
        with torch.no_grad():
            code_idx, _ = vqvae.encode(feat)
        code_idx = code_idx.squeeze(0).cpu()
        tp = code_idx.shape[0] // mcfg.FRAME_SIZE
        if tp > mcfg.MAX_T_PRIME:
            code_idx = code_idx[:mcfg.MAX_T_PRIME * mcfg.FRAME_SIZE]
            tp = mcfg.MAX_T_PRIME
        codes_d[mid] = code_idx
        tp_d[mid] = tp

    torch.save({"codes": codes_d, "t_primes": tp_d}, cache)
    del vqvae
    torch.cuda.empty_cache()
    logger.info(f"Saved VQ cache ({len(codes_d)} samples) → {cache}")
    return cache


class M2TEvalDataset(Dataset):
    """HumanML3D samples for motion-to-text evaluation."""

    def __init__(self, data_root, split, vq_cache_path,
                 w_vectorizer, eval_mean, eval_std,
                 max_motion_length=196, min_motion_length=40,
                 max_text_len=20, unit_length=4):
        self.data_root = data_root
        self.w_vectorizer = w_vectorizer
        self.eval_mean = eval_mean
        self.eval_std = eval_std
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        cache = torch.load(vq_cache_path, weights_only=False)
        self.codes = cache["codes"]
        self.t_primes = cache["t_primes"]

        with open(os.path.join(data_root, f"{split}.txt")) as f:
            all_ids = [l.strip() for l in f if l.strip()]

        self.data = []
        skipped = 0
        for name in all_ids:
            if name not in self.codes or self.t_primes.get(name, 0) < 2:
                skipped += 1
                continue
            txt_path = os.path.join(data_root, "texts", f"{name}.txt")
            mot_path = os.path.join(data_root, "new_joint_vecs", f"{name}.npy")
            if not os.path.exists(txt_path) or not os.path.exists(mot_path):
                skipped += 1
                continue

            motion = np.load(mot_path)
            if np.isnan(motion).any() or np.isinf(motion).any():
                skipped += 1
                continue
            if motion.shape[0] < min_motion_length:
                skipped += 1
                continue

            with open(txt_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            captions, token_lines = [], []
            for line in lines:
                parts = line.split("#")
                cap = parts[0].strip()
                tok = parts[1].strip() if len(parts) > 1 else ""
                if cap:
                    captions.append(cap)
                    token_lines.append(tok)
            if not captions:
                skipped += 1
                continue

            self.data.append({
                "name": name, "captions": captions,
                "token_lines": token_lines, "motion": motion,
            })

        if skipped:
            logger.warning(f"Skipped {skipped} samples")
        logger.info(f"M2TEvalDataset: {len(self.data)} samples from {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        name = item["name"]
        tp = min(self.t_primes[name], mcfg.MAX_T_PRIME)
        codes = self.codes[name][:tp * mcfg.FRAME_SIZE].tolist()

        raw = item["motion"]
        T = raw.shape[0]
        if self.unit_length < 10:
            coin = np.random.choice(["single", "single", "double"])
        else:
            coin = "single"
        m_len = min(T, self.max_motion_length)
        if coin == "double":
            m_len = (m_len // self.unit_length - 1) * self.unit_length
        else:
            m_len = (m_len // self.unit_length) * self.unit_length
        m_len = max(m_len, self.unit_length)
        start = random.randint(0, max(0, T - m_len))
        motion = raw[start:start + m_len].copy()
        motion = (motion - self.eval_mean) / self.eval_std
        if m_len < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - m_len, motion.shape[1]),
                           dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)

        cap_idx = random.randrange(len(item["captions"]))
        tok_str = item["token_lines"][cap_idx]
        tokens = tok_str.split(" ") if tok_str else []

        mtl = self.max_text_len
        if len(tokens) < mtl:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens += ["unk/OTHER"] * (mtl + 2 - sent_len)
        else:
            tokens = tokens[:mtl]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        we, po = [], []
        for tok in tokens:
            try:
                wv, pv = self.w_vectorizer[tok]
            except Exception:
                wv = np.zeros(300, dtype=np.float32)
                pv = np.zeros(15, dtype=np.float32)
            we.append(wv)
            po.append(pv)

        return {
            "name": name,
            "motion_codes": codes,
            "T_prime": tp,
            "captions": item["captions"],
            "all_captions": item["captions"],
            "selected_caption": item["captions"][cap_idx],
            "word_embs": torch.from_numpy(np.stack(we).astype(np.float32)),
            "pos_ohot": torch.from_numpy(np.stack(po).astype(np.float32)),
            "sent_len": sent_len,
            "motion": torch.from_numpy(motion.astype(np.float32)),
            "m_length": m_len,
        }


def m2t_collate(batch):
    return batch


def evaluate(args):
    device = torch.device(args.device)
    os.chdir(REPO_ROOT)

    model, cfg, ckpt_step, tokenizer, backbone = load_model(
        args.checkpoint, args.config, device,
        vq_override=getattr(args, "vq", None))
    logger.info(f"M2T using backbone={backbone}")

    data_root = cfg["data"].get("data_root", "datasets/HumanML3D")
    vq_cache = ensure_vq_cache(data_root, "test", cfg, device)

    do_align = not args.no_align
    eval_wrapper, w_vectorizer = None, None
    eval_mean, eval_std = None, None
    if do_align:
        eval_meta_dir = args.eval_meta or DEFAULT_EVAL_META
        eval_mean = np.load(os.path.join(eval_meta_dir, "mean.npy"))
        eval_std = np.load(os.path.join(eval_meta_dir, "std.npy"))

        eval_opt_path = args.eval_opt or os.path.join(REPO_ROOT,
                                                       DEFAULT_EVAL_OPT)
        eval_ckpt_dir = args.eval_ckpt_dir or os.path.join(REPO_ROOT,
                                                            DEFAULT_EVAL_CKPT_DIR)
        eval_opt = get_opt(eval_opt_path, device, data_root=data_root)
        eval_opt.checkpoints_dir = eval_ckpt_dir
        eval_wrapper = EvaluatorModelWrapper(eval_opt)

        glove_dir = args.glove_dir or os.path.join(REPO_ROOT, DEFAULT_GLOVE_DIR)
        w_vectorizer = WordVectorizer(glove_dir, "our_vab")
    else:
        eval_meta_dir = args.eval_meta or DEFAULT_EVAL_META
        eval_mean = np.load(os.path.join(eval_meta_dir, "mean.npy"))
        eval_std = np.load(os.path.join(eval_meta_dir, "std.npy"))
        glove_dir = args.glove_dir or os.path.join(REPO_ROOT, DEFAULT_GLOVE_DIR)
        w_vectorizer = WordVectorizer(glove_dir, "our_vab")

    dataset = M2TEvalDataset(
        data_root=data_root, split="test", vq_cache_path=vq_cache,
        w_vectorizer=w_vectorizer, eval_mean=eval_mean, eval_std=eval_std,
        max_motion_length=MAX_MOTION_LENGTH, min_motion_length=40,
        max_text_len=20, unit_length=UNIT_LENGTH,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=m2t_collate,
                        drop_last=True)

    logger.info(f"Test samples: {len(dataset)}, Batches: {len(loader)}")
    logger.info(f"Generation: steps={args.num_steps}, cfg={args.cfg_scale}")

    all_results = []

    for rep in range(args.num_repeat):
        logger.info(f"═══ Repetition {rep + 1}/{args.num_repeat} ═══")

        all_preds = []
        all_refs_single = []      # MotionGPT3-style single ref (one per sample)
        all_refs_multi = []
        gen_text_embs_list = []
        gt_text_embs_list = []
        motion_embs_list = []
        n_gen, n_fail = 0, 0
        sample_log = []

        for batch in tqdm(loader, desc=f"Rep {rep + 1}", leave=False):
            bs = len(batch)

            pred_captions = []
            for item in batch:
                try:
                    tids = generate_m2t(
                        model, item["motion_codes"], item["T_prime"],
                        num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                        device=str(device),
                        corrector_steps=args.corrector_steps,
                        corrector_freq=args.corrector_freq,
                        corrector_time=args.corrector_time_list,
                        max_text_len=args.max_text_len,
                        temperature=args.temperature,
                    )
                    cap = tokenizer.decode(tids, skip_special_tokens=True)
                    if not cap.strip():
                        cap = "a person is standing."
                        n_fail += 1
                except Exception as e:
                    if n_fail < 5:
                        logger.warning(f"M2T failed [{item['name']}]: {e}")
                    cap = "a person is standing."
                    n_fail += 1
                pred_captions.append(cap)
                n_gen += 1

            for item, pred in zip(batch, pred_captions):
                all_preds.append(pred)
                all_refs_single.append(item["selected_caption"])
                all_refs_multi.append(item["all_captions"])
                if len(sample_log) < 50:
                    sample_log.append({
                        "name": item["name"],
                        "prediction": pred,
                        "selected_reference": item["selected_caption"],
                        "all_references": item["all_captions"],
                    })

            if do_align:
                gt_we = torch.stack([b["word_embs"] for b in batch])
                gt_po = torch.stack([b["pos_ohot"] for b in batch])
                gt_sl = torch.tensor([b["sent_len"] for b in batch],
                                     dtype=torch.long)
                motions = torch.stack([b["motion"] for b in batch])
                m_lens = torch.tensor([b["m_length"] for b in batch],
                                      dtype=torch.long)

                gt_te, mo_e = eval_wrapper.get_co_embeddings(
                    gt_we, gt_po, gt_sl, motions, m_lens.to(device))
                gt_text_embs_list.append(gt_te.cpu().numpy())
                motion_embs_list.append(mo_e.cpu().numpy())

                gen_we, gen_po, gen_sl = [], [], []
                for pred in pred_captions:
                    we, po, sl = caption_to_evaluator_input(
                        pred, w_vectorizer, max_text_len=20)
                    gen_we.append(we)
                    gen_po.append(po)
                    gen_sl.append(sl)
                gen_we = torch.stack(gen_we)
                gen_po = torch.stack(gen_po)
                gen_sl = torch.tensor(gen_sl, dtype=torch.long)

                gen_te, _ = eval_wrapper.get_co_embeddings(
                    gen_we, gen_po, gen_sl, motions, m_lens.to(device))
                gen_text_embs_list.append(gen_te.cpu().numpy())

        logger.info(f"Generated: {n_gen}, Failed: {n_fail}")

        refs_for_nlg = all_refs_multi
        metrics = {}
        logger.info("Computing BLEU …")
        metrics.update(compute_bleu(refs_for_nlg, all_preds))
        logger.info("Computing ROUGE-L …")
        metrics.update(compute_rouge_l(refs_for_nlg, all_preds))
        logger.info("Computing CIDEr …")
        metrics.update(compute_cider(refs_for_nlg, all_preds))
        if not args.no_bert_score:
            logger.info("Computing BERTScore …")
            metrics.update(compute_bert_score(
                all_refs_single, all_preds, device=str(device),
                batch_size=args.bert_score_batch_size,
                isolate=not args.bert_score_inproc))

        if do_align:
            logger.info("Computing R-Precision & MM-Dist …")
            gen_all = np.concatenate(gen_text_embs_list)
            gt_all = np.concatenate(gt_text_embs_list)
            mo_all = np.concatenate(motion_embs_list)
            align = compute_alignment_metrics(gen_all, gt_all, mo_all)
            metrics.update(align)

        metrics["generation_failures"] = n_fail
        all_results.append(metrics)

        logger.info(
            f"  Bleu_1={metrics.get('Bleu_1', 0):.4f}  "
            f"Bleu_4={metrics.get('Bleu_4', 0):.4f}  "
            f"Rouge_L={metrics.get('Rouge_L', 0):.4f}  "
            f"CIDEr={metrics.get('CIDEr', 0):.4f}  "
            f"BERTScore={metrics.get('BERTScore', 0):.4f}"
        )
        if do_align:
            logger.info(
                f"  R1={metrics['R1']:.4f}  R2={metrics['R2']:.4f}  "
                f"R3={metrics['R3']:.4f}  MM-Dist={metrics['MM-Dist']:.4f}"
            )
            logger.info(
                f"  [GT] R1={metrics['gt_R1']:.4f}  "
                f"R2={metrics['gt_R2']:.4f}  R3={metrics['gt_R3']:.4f}  "
                f"MM-Dist={metrics['gt_MM-Dist']:.4f}"
            )

    logger.info("=" * 80)
    nlp_keys = [k for k in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4",
                             "Rouge_L", "CIDEr", "BERTScore"]
                if k in all_results[0]]
    align_keys = [k for k in ["R1", "R2", "R3", "MM-Dist",
                               "gt_R1", "gt_R2", "gt_R3", "gt_MM-Dist"]
                  if k in all_results[0]]
    summary_keys = nlp_keys + align_keys

    def _agg(vals):
        v = np.asarray(vals, dtype=float)
        n = int(np.isfinite(v).sum())
        if n == 0:
            return float("nan"), float("nan"), float("nan")
        mean = float(np.nanmean(v))
        std = float(np.nanstd(v))
        ci = std * 1.96 / max(np.sqrt(n), 1.0)
        return mean, std, ci

    if args.num_repeat > 1:
        logger.info(f"Summary over {args.num_repeat} repetitions:")
        for key in summary_keys:
            mean, _, ci = _agg([r[key] for r in all_results])
            logger.info(f"  {key:>12s}: {mean:.4f} ± {ci:.4f}")
    else:
        m = all_results[0]
        parts = [f"{k}={m[k]:.4f}" for k in summary_keys]
        logger.info("  ".join(parts))
    logger.info("=" * 80)

    out_dir = os.path.dirname(args.checkpoint)
    ts = datetime.now().strftime("%d-%H-%M-%S")
    result_path = os.path.join(out_dir, f"eval_m2t_step{ckpt_step}_{ts}.json")

    if args.output_json:
        result_path = args.output_json
        os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    summary_dict = {}
    for key in summary_keys:
        mean, std, ci = _agg([r[key] for r in all_results])
        summary_dict[key] = {"mean": mean, "std": std, "ci95": ci}

    with open(result_path, "w") as f:
        json.dump({
            "config": {
                "protocol": "humanml3d-4208",
                "sample": args.sample,
                "seed": args.seed,
                "num_workers": args.num_workers,
                "checkpoint": args.checkpoint,
                "num_steps": args.num_steps,
                "cfg_scale": args.cfg_scale,
                "num_repeat": args.num_repeat,
                "batch_size": args.batch_size,
                "corrector_steps": args.corrector_steps,
                "corrector_freq": args.corrector_freq,
                "corrector_time": args.corrector_time,
                "max_text_len": args.max_text_len,
                "temperature": args.temperature,
            },
            "results": all_results,
            "summary": summary_dict,
            "samples": sample_log,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved → {result_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="M2T evaluation")
    p.add_argument("--sample", type=int, choices=[4208], default=4208)
    p.add_argument("--output-json", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/bimogen.yaml")
    p.add_argument("--device", default="cuda")

    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for token selection. 0=argmax. "
                        "Try 0.3/0.5/0.7 to add diversity (helps R-Precision).")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-repeat", type=int, default=1)

    p.add_argument("--eval-opt", default=None)
    p.add_argument("--eval-ckpt-dir", default=None)
    p.add_argument("--eval-meta", default=None)
    p.add_argument("--glove-dir", default=None)

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--vq", choices=["frame"], default=None)

    p.add_argument("--corrector-steps", type=int, default=0)
    p.add_argument("--corrector-freq", type=int, default=1)
    p.add_argument("--corrector-time", type=str, default=None,
                   help="Comma-separated 1-indexed steps for corrector "
                        "(e.g. '1,2,3'). Overrides --corrector-freq.")

    p.add_argument("--max-text-len", type=int, default=None,
                   help="Number of text content slots between SOT and EOT at "
                        "inference time (default uses MAX_TEXT_CONTENT=50). "
                        "Try 20/25/30 to match HumanML3D caption length.")

    p.add_argument("--no-align", action="store_true",
                   help="Skip alignment metrics (R-Precision, MM-Dist)")
    p.add_argument("--no-bert-score", action="store_true",
                   help="Skip BERTScore computation")
    p.add_argument("--bert-score-batch-size", type=int, default=64,
                   help="bert-score internal batch size; lower it (e.g. 16) "
                        "if you hit OOM or instability on RoBERTa-large.")
    p.add_argument("--bert-score-inproc", action="store_true",
                   help="Run bert-score in the main process (faster, but a "
                        "RoBERTa segfault will kill the whole eval).")

    args = p.parse_args()

    if args.corrector_time is not None:
        args.corrector_time_list = [int(x) for x in args.corrector_time.split(",")]
    else:
        args.corrector_time_list = None

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    print(args)
    evaluate(args)
