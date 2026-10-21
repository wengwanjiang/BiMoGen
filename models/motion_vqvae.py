"""Self-contained MotionGPT VQ-VAE for frame-wise motion tokenization.

Architecture: 1D-CNN encoder/decoder + EMA-reset vector quantizer.
Config (from MotionGPT vq/default.yaml):
    code_num=512, code_dim=512, down_t=2, stride_t=2 -> 4x temporal downsampling
    width=512, depth=3, dilation_growth_rate=3, activation='relu', norm=None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple
from torch import Tensor
# Building blocks

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='relu', norm=None):
        super().__init__()
        padding = dilation
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(32, n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(32, n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.use_ln = (norm == "LN")

        act_map = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}
        act_cls = act_map.get(activation, nn.ReLU)
        self.activation1 = act_cls()
        self.activation2 = act_cls()
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)

    def forward(self, x):
        x_orig = x
        if self.use_ln:
            x = self.activation1(self.norm1(x.transpose(-2, -1)).transpose(-2, -1))
        else:
            x = self.activation1(self.norm1(x))
        x = self.conv1(x)
        if self.use_ln:
            x = self.activation2(self.norm2(x.transpose(-2, -1)).transpose(-2, -1))
        else:
            x = self.activation2(self.norm2(x))
        x = self.conv2(x)
        return x + x_orig


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1,
                 reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        blocks = [
            ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** d,
                           activation=activation, norm=norm)
            for d in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
# Quantizer

class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu=0.99):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.register_buffer('codebook', torch.zeros(nb_code, code_dim))


    def quantize(self, x):
        k_w = self.codebook.t()
        dist = (torch.sum(x ** 2, dim=-1, keepdim=True)
                - 2 * torch.matmul(x, k_w)
                + torch.sum(k_w ** 2, dim=0, keepdim=True))
        _, code_idx = torch.min(dist, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        return F.embedding(code_idx, self.codebook)
# Encoder / Decoder

class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t,
                 width, depth, dilation_growth_rate, activation='relu', norm=None):
        super().__init__()
        blocks = []
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate,
                         activation=activation, norm=norm),
            ))
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t,
                 width, depth, dilation_growth_rate, activation='relu', norm=None):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate,
                         reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, width, 3, 1, 1),
            ))
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
# VQVae

class VQVae(nn.Module):
    def __init__(self, nfeats=263, code_num=512, code_dim=512,
                 output_emb_width=512, down_t=2, stride_t=2, width=512,
                 depth=3, dilation_growth_rate=3, activation='relu', norm=None,
                 quantizer='ema_reset', **kwargs):
        super().__init__()
        self.code_dim = code_dim
        self.code_num = code_num
        self.encoder = Encoder(nfeats, output_emb_width, down_t, stride_t,
                               width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)
        self.decoder = Decoder(nfeats, output_emb_width, down_t, stride_t,
                               width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)
        self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)

    def encode(self, features: Tensor) -> Tuple[Tensor, None]:
        """Encode motion features to discrete code indices.

        Args:
            features: (B, T, nfeats) raw (z-normed) motion features.
        Returns:
            code_idx: (B, T') integer indices in [0, code_num).
        """
        N, T, _ = features.shape
        x = features.permute(0, 2, 1)            # (B, C, T)
        x = self.encoder(x)                       # (B, dim, T')
        x = x.permute(0, 2, 1).contiguous()       # (B, T', dim)
        x = x.view(-1, x.shape[-1])               # (B*T', dim)
        code_idx = self.quantizer.quantize(x)      # (B*T',)
        code_idx = code_idx.view(N, -1)            # (B, T')
        return code_idx, None

    def decode(self, z: Tensor) -> Tensor:
        """Decode code indices back to motion features.

        Args:
            z: integer code indices, shape (B, T') or (T',).
        Returns:
            motion features (B, T_out, nfeats).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B, L = z.shape
        x_d = self.quantizer.dequantize(z.reshape(-1))   # (B*L, dim)
        x_d = x_d.view(B, L, self.code_dim)              # (B, L, dim)
        x_d = x_d.permute(0, 2, 1).contiguous()          # (B, dim, L)
        x_out = self.decoder(x_d)                         # (B, nfeats, T_out)
        x_out = x_out.permute(0, 2, 1)                    # (B, T_out, nfeats)
        return x_out
# Checkpoint loading

def load_motion_vqvae(
    checkpoint_path: str = "deps/motiongpt/motiongpt_s3_h3d.tar",
    nfeats: int = 263,
    device: str = "cpu",
) -> VQVae:
    """Load the pre-trained MotionGPT VQ-VAE from a full-model checkpoint.

    The checkpoint stores the full MotionGPT model; VQ-VAE weights are
    prefixed with 'motion_vae.' or 'vae.'.
    """
    model = VQVae(nfeats=nfeats, code_num=512, code_dim=512,
                  output_emb_width=512, down_t=2, stride_t=2,
                  width=512, depth=3, dilation_growth_rate=3,
                  activation='relu', norm=None)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    vae_dict = OrderedDict()
    for key, value in state_dict.items():
        for prefix in ('motion_vae.', 'vae.'):
            if key.startswith(prefix):
                vae_dict[key[len(prefix):]] = value
                break
    if not vae_dict:
        vae_dict = state_dict  # Also accept a standalone VQ-VAE state dict.
    model.load_state_dict(vae_dict, strict=True)
    return model.eval().requires_grad_(False).to(device)
