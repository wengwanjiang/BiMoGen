"""T2M evaluation: FID, retrieval, matching distance, and diversity."""
import argparse
import os
import sys
import random
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import yaml

import models.configuration as mcfg
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from utils.get_eval_option import get_opt
from evaluation.metrics.utils import (
    euclidean_distance_matrix, calculate_top_k,
    calculate_activation_statistics_np, calculate_frechet_distance_np,
    calculate_diversity_np,
)
from inference.generate import generate_t2m

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_t2m")

DEFAULT_EVAL_OPT = "deps/t2m/t2m/Comp_v6_KLD005/opt.txt"
DEFAULT_EVAL_CKPT_DIR = "deps/t2m"
DEFAULT_GLOVE_DIR = "deps/glove"
DEFAULT_EVAL_META = "deps/t2m/t2m/Comp_v6_KLD005/meta"

MAX_MOTION_LENGTH = 196
MIN_MOTION_LENGTH = 40
UNIT_LENGTH = 4
FPS = 20


class T2MEvalDataset(Dataset):
    """HumanML3D samples for text-to-motion evaluation."""

    def __init__(self, data_root, split, w_vectorizer, eval_mean, eval_std,
                 max_motion_length=196, min_motion_length=40,
                 max_text_len=20, unit_length=4):
        self.data_root = data_root
        self.w_vectorizer = w_vectorizer
        self.eval_mean = eval_mean
        self.eval_std = eval_std
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        split_path = os.path.join(data_root, f"{split}.txt")
        with open(split_path) as f:
            all_ids = [line.strip() for line in f if line.strip()]

        self.data = []
        skipped = 0
        for name in all_ids:
            motion_path = os.path.join(data_root, "new_joint_vecs", f"{name}.npy")
            text_path = os.path.join(data_root, "texts", f"{name}.txt")
            if not os.path.exists(motion_path) or not os.path.exists(text_path):
                skipped += 1
                continue
            motion = np.load(motion_path)
            if np.isnan(motion).any() or np.isinf(motion).any():
                skipped += 1
                continue
            T = motion.shape[0]
            if T < min_motion_length:
                skipped += 1
                continue
            self.data.append((name, motion))
        if skipped:
            logger.warning(f"Skipped {skipped} samples (missing/corrupt/too short)")
        logger.info(f"T2MEvalDataset: {len(self.data)} samples from {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, motion_raw = self.data[idx]
        T = motion_raw.shape[0]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        m_length = min(T, self.max_motion_length)
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        m_length = max(m_length, self.unit_length)

        start = random.randint(0, max(0, T - m_length))
        motion = motion_raw[start:start + m_length].copy()

        motion = (motion - self.eval_mean) / self.eval_std

        if m_length < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - m_length, motion.shape[1]),
                           dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)

        text_path = os.path.join(self.data_root, "texts", f"{name}.txt")
        with open(text_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        line = random.choice(lines)
        parts = line.split("#")
        caption = parts[0]
        tokens = parts[1].split(" ") if len(parts) > 1 else []

        max_tl = self.max_text_len  # 20
        if len(tokens) < max_tl:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_tl + 2 - sent_len)
        else:
            tokens = tokens[:max_tl]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        word_embs_list = []
        pos_ohot_list = []
        for token in tokens:
            try:
                w_vec, p_vec = self.w_vectorizer[token]
            except Exception:
                w_vec = np.zeros(300, dtype=np.float32)
                p_vec = np.zeros(15, dtype=np.float32)
            word_embs_list.append(w_vec)
            pos_ohot_list.append(p_vec)

        word_embs = np.stack(word_embs_list, axis=0).astype(np.float32)
        pos_ohot = np.stack(pos_ohot_list, axis=0).astype(np.float32)

        return (torch.from_numpy(word_embs),
                torch.from_numpy(pos_ohot),
                caption, sent_len,
                torch.from_numpy(motion.astype(np.float32)),
                m_length, name)


def t2m_eval_collate(batch):
    word_embs = torch.stack([b[0] for b in batch])
    pos_ohot = torch.stack([b[1] for b in batch])
    captions = [b[2] for b in batch]
    sent_lens = torch.tensor([b[3] for b in batch], dtype=torch.long)
    motions = torch.stack([b[4] for b in batch])
    m_lengths = torch.tensor([b[5] for b in batch], dtype=torch.long)
    names = [b[6] for b in batch]
    return word_embs, pos_ohot, captions, sent_lens, motions, m_lengths, names


from models.loading import load_model


def decode_motion_tokens(tokens, vqvae, device):
    """Decode motion token IDs -> (1, T_raw, 263) in VQ-VAE normalization."""
    codes = []
    for t in tokens:
        c = t - mcfg.MOTION_CODE_START
        c = max(0, min(mcfg.CODEBOOK_SIZE - 1, c))
        codes.append(c)
    if not codes:
        return None, 0
    code_t = torch.tensor(codes, dtype=torch.long, device=device)
    with torch.no_grad():
        motion = vqvae.decode(code_t)
    return motion, motion.shape[1]


def renorm_motion(motion_tensor, vqvae_mean, vqvae_std, eval_mean, eval_std):
    """Re-normalize motion features from VQ-VAE space to evaluator space.

    motion_tensor: (B, T, 263) in VQ-VAE normalization
    Returns: (B, T, 263) in evaluator normalization
    """
    raw = motion_tensor * vqvae_std + vqvae_mean
    return (raw - eval_mean) / eval_std


def compute_metrics(all_text_embs, all_gt_embs, all_gen_embs,
                    top_k=3, R_size=32, diversity_times=300):
    """Compute FID, R-Precision, MM-Dist, Diversity from embedding arrays.

    All inputs are numpy arrays of shape (N, D).
    """
    N = len(all_text_embs)
    assert N == len(all_gt_embs) == len(all_gen_embs)

    shuffle_idx = np.random.permutation(N)
    all_text = all_text_embs[shuffle_idx]
    all_gen = all_gen_embs[shuffle_idx]
    all_gt = all_gt_embs[shuffle_idx]

    R_count = (N // R_size) * R_size
    matching_score = 0.0
    gt_matching_score = 0.0
    top_k_counts = np.zeros(top_k)
    gt_top_k_counts = np.zeros(top_k)

    for i in range(N // R_size):
        s, e = i * R_size, (i + 1) * R_size
        t_emb = torch.from_numpy(all_text[s:e])
        g_emb = torch.from_numpy(all_gen[s:e])
        gt_emb = torch.from_numpy(all_gt[s:e])

        dist_gen = euclidean_distance_matrix(t_emb, g_emb).nan_to_num()
        matching_score += dist_gen.trace().item()
        argsorted = torch.argsort(dist_gen, dim=1)
        top_k_counts += calculate_top_k(argsorted, top_k=top_k).sum(axis=0).numpy()

        dist_gt = euclidean_distance_matrix(t_emb, gt_emb).nan_to_num()
        gt_matching_score += dist_gt.trace().item()
        argsorted_gt = torch.argsort(dist_gt, dim=1)
        gt_top_k_counts += calculate_top_k(argsorted_gt, top_k=top_k).sum(axis=0).numpy()

    r_precision = top_k_counts / R_count
    gt_r_precision = gt_top_k_counts / R_count
    mm_dist = matching_score / R_count
    gt_mm_dist = gt_matching_score / R_count

    gen_mu, gen_cov = calculate_activation_statistics_np(all_gen)
    gt_mu, gt_cov = calculate_activation_statistics_np(all_gt)
    fid = calculate_frechet_distance_np(gt_mu, gt_cov, gen_mu, gen_cov)

    n_div = min(diversity_times, N - 1)
    diversity = calculate_diversity_np(all_gen, n_div)
    gt_diversity = calculate_diversity_np(all_gt, n_div)

    return {
        "FID": float(fid),
        "R1": float(r_precision[0]),
        "R2": float(r_precision[1]),
        "R3": float(r_precision[2]),
        "MM-Dist": float(mm_dist),
        "Diversity": float(diversity),
        "gt_R1": float(gt_r_precision[0]),
        "gt_R2": float(gt_r_precision[1]),
        "gt_R3": float(gt_r_precision[2]),
        "gt_MM-Dist": float(gt_mm_dist),
        "gt_Diversity": float(gt_diversity),
    }


def evaluate(args):
    device = torch.device(args.device)
    os.chdir(REPO_ROOT)

    model, cfg, ckpt_step, tokenizer, backbone = load_model(
        args.checkpoint, args.config, device,
        vq_override=getattr(args, 'vq', None))
    logger.info(f"T2M using backbone={backbone}")

    vq_ckpt = cfg["vqvae"]["checkpoint"]
    from models.motion_vqvae import load_motion_vqvae
    vqvae = load_motion_vqvae(vq_ckpt, device=str(device))
    vqvae.eval()

    vqvae_mean = np.load(cfg["data"]["mean_path"])  # (263,)
    vqvae_std = np.load(cfg["data"]["std_path"])

    eval_meta_dir = args.eval_meta or DEFAULT_EVAL_META
    eval_mean = np.load(os.path.join(eval_meta_dir, "mean.npy"))
    eval_std = np.load(os.path.join(eval_meta_dir, "std.npy"))

    vqvae_mean_t = torch.from_numpy(vqvae_mean).float().to(device)
    vqvae_std_t = torch.from_numpy(vqvae_std).float().to(device)
    eval_mean_t = torch.from_numpy(eval_mean).float().to(device)
    eval_std_t = torch.from_numpy(eval_std).float().to(device)

    eval_opt_path = args.eval_opt or os.path.join(REPO_ROOT, DEFAULT_EVAL_OPT)
    eval_ckpt_dir = args.eval_ckpt_dir or os.path.join(REPO_ROOT, DEFAULT_EVAL_CKPT_DIR)
    data_root = cfg["data"].get("data_root", "datasets/HumanML3D")

    eval_opt = get_opt(eval_opt_path, device, data_root=data_root)
    eval_opt.checkpoints_dir = eval_ckpt_dir
    eval_wrapper = EvaluatorModelWrapper(eval_opt)

    glove_dir = args.glove_dir or os.path.join(REPO_ROOT, DEFAULT_GLOVE_DIR)
    w_vectorizer = WordVectorizer(glove_dir, "our_vab")

    dataset = T2MEvalDataset(
        data_root=data_root, split="test",
        w_vectorizer=w_vectorizer,
        eval_mean=eval_mean, eval_std=eval_std,
        max_motion_length=MAX_MOTION_LENGTH,
        min_motion_length=MIN_MOTION_LENGTH,
        max_text_len=eval_opt.max_text_len,
        unit_length=UNIT_LENGTH,

    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=t2m_eval_collate,
        drop_last=True,
    )

    logger.info(f"Test samples: {len(dataset)}")
    logger.info(f"Batch size: {args.batch_size}, Batches: {len(loader)}")
    logger.info(f"Generation: steps={args.num_steps}, cfg={args.cfg_scale}, temp={args.temperature}")
    logger.info(f"Repetitions: {args.num_repeat}, true_len={args.true_len}")

    all_results = []
    corrector_time_list = getattr(args, 'corrector_time_list', None)

    for rep in range(args.num_repeat):
        logger.info(f"═══ Repetition {rep + 1}/{args.num_repeat} ═══")

        text_embs_all = []
        gt_motion_embs_all = []
        gen_motion_embs_all = []
        n_generated = 0
        n_failed = 0

        for batch in tqdm(loader, desc=f"Rep {rep + 1}", leave=False):
            word_embs, pos_ohot, captions, sent_lens, gt_motions, m_lengths, names = batch
            bs = gt_motions.shape[0]

            gt_dev = gt_motions.to(device)
            text_emb, gt_emb = eval_wrapper.get_co_embeddings(
                word_embs, pos_ohot, sent_lens, gt_dev, m_lengths.to(device))

            gen_motions = torch.zeros_like(gt_dev)
            gen_m_lengths = torch.zeros(bs, dtype=torch.long)

            for i in range(bs):
                text_ids = tokenizer(
                    captions[i], add_special_tokens=False,
                    truncation=True, max_length=mcfg.MAX_TEXT_CONTENT,
                )["input_ids"]

                try:
                    gt_T_prime = m_lengths[i].item() // mcfg.TEMPORAL_DOWNSAMPLE
                    motion_tokens = generate_t2m(
                        model, text_ids,
                        num_steps=args.num_steps,
                        cfg_scale=args.cfg_scale,
                        target_T_prime=gt_T_prime,
                        device=str(device),
                        corrector_steps=args.corrector_steps,
                        corrector_freq=args.corrector_freq,
                        corrector_time=corrector_time_list,
                        temperature=args.temperature,
                    )
                    dec, T_raw = decode_motion_tokens(motion_tokens, vqvae, device)
                    if dec is not None and T_raw > 0:
                        T_use = min(T_raw, MAX_MOTION_LENGTH)
                        T_use = (T_use // UNIT_LENGTH) * UNIT_LENGTH
                        T_use = max(T_use, UNIT_LENGTH)
                        gen_feat = renorm_motion(
                            dec[:, :T_use], vqvae_mean_t, vqvae_std_t,
                            eval_mean_t, eval_std_t)
                        gen_motions[i, :T_use] = gen_feat[0]
                        gen_m_lengths[i] = T_use
                    else:
                        gen_m_lengths[i] = m_lengths[i]
                        n_failed += 1
                except Exception as e:
                    if n_failed < 5:
                        logger.warning(f"Generation failed [{names[i]}]: {e}")
                    gen_m_lengths[i] = m_lengths[i]
                    n_failed += 1

                n_generated += 1

            _, gen_emb = eval_wrapper.get_co_embeddings(
                word_embs, pos_ohot, sent_lens, gen_motions, gen_m_lengths.to(device))

            text_embs_all.append(text_emb.cpu().numpy())
            gt_motion_embs_all.append(gt_emb.cpu().numpy())
            gen_motion_embs_all.append(gen_emb.cpu().numpy())

        all_text = np.concatenate(text_embs_all, axis=0)
        all_gt = np.concatenate(gt_motion_embs_all, axis=0)
        all_gen = np.concatenate(gen_motion_embs_all, axis=0)

        logger.info(f"Generated: {n_generated}, Failed: {n_failed}")

        metrics = compute_metrics(all_text, all_gt, all_gen,
                                  top_k=3, R_size=32, diversity_times=300)
        metrics["generation_failures"] = n_failed
        all_results.append(metrics)

        logger.info(
            f"  FID={metrics['FID']:.4f}  "
            f"R1={metrics['R1']:.4f}  R2={metrics['R2']:.4f}  R3={metrics['R3']:.4f}  "
            f"MM-Dist={metrics['MM-Dist']:.4f}  Div={metrics['Diversity']:.4f}"
        )
        logger.info(
            f"  [GT] R1={metrics['gt_R1']:.4f}  R2={metrics['gt_R2']:.4f}  "
            f"R3={metrics['gt_R3']:.4f}  MM-Dist={metrics['gt_MM-Dist']:.4f}  "
            f"Div={metrics['gt_Diversity']:.4f}"
        )

    logger.info("=" * 80)
    if args.num_repeat > 1:
        logger.info(f"Summary over {args.num_repeat} repetitions (mean +/- 95% CI):")
        for key in ["FID", "R1", "R2", "R3", "MM-Dist", "Diversity"]:
            vals = np.array([r[key] for r in all_results])
            mean = vals.mean()
            ci = vals.std() * 1.96 / np.sqrt(len(vals))
            logger.info(f"  {key:>10s}: {mean:.4f} +/- {ci:.4f}")
        logger.info("GT reference:")
        for key in ["gt_R1", "gt_R2", "gt_R3", "gt_MM-Dist", "gt_Diversity"]:
            vals = np.array([r[key] for r in all_results])
            mean = vals.mean()
            ci = vals.std() * 1.96 / np.sqrt(len(vals))
            logger.info(f"  {key:>12s}: {mean:.4f} +/- {ci:.4f}")
    else:
        m = all_results[0]
        logger.info(f"FID={m['FID']:.4f}  R1={m['R1']:.4f}  R2={m['R2']:.4f}  "
                     f"R3={m['R3']:.4f}  MM-Dist={m['MM-Dist']:.4f}  "
                     f"Div={m['Diversity']:.4f}")
    logger.info("=" * 80)

    out_dir = os.path.dirname(args.checkpoint)
    timestamp = datetime.now().strftime("%d-%H-%M-%S")
    result_path = os.path.join(out_dir, f"eval_t2m_step{ckpt_step}_{timestamp}.json")

    if args.output_json:
        result_path = args.output_json
        os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    try:
        import json
        with open(result_path, "w") as f:
            json.dump({
                "config": {
                    "checkpoint": args.checkpoint,
                    "num_steps": args.num_steps,
                    "cfg_scale": args.cfg_scale,
                    "temperature": args.temperature,
                    "num_repeat": args.num_repeat,
                    "batch_size": args.batch_size,
                    "true_len": args.true_len,
                    "sample": args.sample,
                    "protocol": "humanml3d-4208",
                    "seed": args.seed,
                    "num_workers": args.num_workers,
                    "corrector_freq": args.corrector_freq,
                    "corrector_steps": args.corrector_steps,
                    "corrector_time": args.corrector_time,
                },
                "results": all_results,
                "summary": {
                    key: {
                        "mean": float(np.mean([r[key] for r in all_results])),
                        "std": float(np.std([r[key] for r in all_results])),
                        "ci95": float(np.std([r[key] for r in all_results]) * 1.96
                                      / np.sqrt(len(all_results))),
                    }
                    for key in ["FID", "R1", "R2", "R3", "MM-Dist", "Diversity"]
                },
            }, f, indent=2)
        logger.info(f"Results saved to {result_path}")
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2M evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint dir (e.g. output/xxx/checkpoint-50000)")
    parser.add_argument("--config", type=str, default="configs/bimogen.yaml")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--true-len", action="store_true",
                        help="Use GT motion length")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-repeat", type=int, default=1,
                        help="Number of repetitions for mean+/-CI")
    parser.add_argument("--sample", type=int, choices=[4208], default=4208)
    parser.add_argument("--output-json", default=None)

    parser.add_argument("--eval-opt", type=str, default=None)
    parser.add_argument("--eval-ckpt-dir", type=str, default=None)
    parser.add_argument("--eval-meta", type=str, default=None,
                        help="Dir with evaluator mean.npy/std.npy")
    parser.add_argument("--glove-dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--corrector-steps", type=int, default=0,
                        help="Self-correction: max corrector steps per loop (0=disabled)")
    parser.add_argument("--corrector-freq", type=int, default=1,
                        help="Self-correction: run corrector every N unmasking steps")
    parser.add_argument("--corrector-time", type=str, default=None,
                        help="Self-correction: comma-separated 1-indexed steps to apply corrector "
                             "(e.g. '1,2,3' = correct at steps 1,2,3 only). "
                             "Overrides --corrector-freq when set.")
    parser.add_argument("--vq", type=str, choices=["frame"], default=None,
                        help="VQ-VAE mode override (auto-detected from checkpoint if not set)")

    args = parser.parse_args()

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
