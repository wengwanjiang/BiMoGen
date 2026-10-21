import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import models.configuration as mcfg

DEFAULT_MEAN_STD_DIR = "deps/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"

class MotionTextDataset(Dataset):

    def __init__(self, data_root, split, tokenizer, vqvae=None,
                 mean_path=None, std_path=None, max_text_len=None,
                 device="cpu"):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len or mcfg.MAX_TEXT_CONTENT

        mean_path = mean_path or os.path.join(DEFAULT_MEAN_STD_DIR, "mean.npy")
        std_path = std_path or os.path.join(DEFAULT_MEAN_STD_DIR, "std.npy")
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        with open(os.path.join(data_root, f"{split}.txt")) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.texts = {}
        for mid in self.ids:
            txt_path = os.path.join(data_root, "texts", f"{mid}.txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path) as f:
                captions = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    caption = line.split("#")[0].strip()
                    if caption:
                        captions.append(caption)
                if captions:
                    self.texts[mid] = captions

        self.ids = [mid for mid in self.ids if mid in self.texts]

        cache_prefix = "motiongpt"
        cache_file = os.path.join(data_root, f"{cache_prefix}_vq_codes_{split}.pt")
        if os.path.exists(cache_file):
            cache = torch.load(cache_file, weights_only=False)
        else:
            if vqvae is None:
                raise RuntimeError(
                    f"No cache at {cache_file} and no VQ-VAE provided. "
                    f"Pass vqvae= to build the cache on first run."
                )
            cache = self._encode_all(data_root, vqvae, device)
            torch.save(cache, cache_file)

        self.codes = cache["codes"]
        self.t_primes = cache["t_primes"]

        before = len(self.ids)
        self.ids = [mid for mid in self.ids
                    if mid in self.codes and self.t_primes.get(mid, 0) >= mcfg.MIN_T_PRIME]
        n_dropped = before - len(self.ids)
        if n_dropped > 0:
            print(f"[dataset] Filtered {n_dropped} samples with T' < {mcfg.MIN_T_PRIME} "
                  f"({len(self.ids)} remaining)")

    @torch.no_grad()
    def _encode_all(self, data_root, vqvae, device):
        vqvae = vqvae.to(device).eval()
        codes_dict, tp_dict = {}, {}
        motion_dir = os.path.join(data_root, "new_joint_vecs")
        frame_size = mcfg.FRAME_SIZE

        desc = "Encoding motions with MotionGPT VQ-VAE"
        for mid in tqdm(self.ids, desc=desc):
            npy_path = os.path.join(motion_dir, f"{mid}.npy")
            if not os.path.exists(npy_path):
                continue
            motion = np.load(npy_path)  # (T, 263)

            T = min(motion.shape[0], mcfg.MAX_MOTION_FRAMES)
            if T < mcfg.MIN_T_PRIME * mcfg.TEMPORAL_DOWNSAMPLE:
                continue
            motion = motion[:T]
            motion = motion[:T // mcfg.TEMPORAL_DOWNSAMPLE * mcfg.TEMPORAL_DOWNSAMPLE, :]

            normed = (motion - self.mean) / self.std
            feat = torch.from_numpy(normed).float().unsqueeze(0).to(device)
            code_idx, _ = vqvae.encode(feat)
            code_idx = code_idx.squeeze(0).cpu()

            tp = code_idx.shape[0] // frame_size

            if tp > mcfg.MAX_T_PRIME:
                code_idx = code_idx[:mcfg.MAX_T_PRIME * frame_size]
                tp = mcfg.MAX_T_PRIME

            codes_dict[mid] = code_idx
            tp_dict[mid] = tp

        vqvae.cpu()
        return {"codes": codes_dict, "t_primes": tp_dict}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        mid = self.ids[idx]
        text = random.choice(self.texts[mid])
        tokens = self.tokenizer(
            text, truncation=True, max_length=self.max_text_len,
            add_special_tokens=False, return_attention_mask=False,
        )
        tp = min(self.t_primes[mid], mcfg.MAX_T_PRIME)
        codes = self.codes[mid][:tp * mcfg.FRAME_SIZE].tolist()
        return {
            "text_tokens": tokens["input_ids"],
            "motion_codes": codes,
            "T_prime": tp,
        }
