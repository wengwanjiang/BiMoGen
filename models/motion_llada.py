"""Bidirectional LLaDA with masked-denoising and self-correction losses."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modeling_llada import LLaDAModelLM
from models.configuration_llada import LLaDAConfig
import models.configuration as mcfg

log = logging.getLogger(__name__)

class MotionLLaDA(nn.Module):
    def __init__(self, pad_loss_weight=0.1, d_model=1024, n_heads=16,
                 n_layers=20, mlp_hidden_size=3072, dropout=0.0,
                 max_position=512):
        super().__init__()
        self.pad_loss_weight = pad_loss_weight
        self.hidden_size = d_model

        vocab_size = mcfg.TOTAL_VOCAB
        embedding_size = (vocab_size + 127) // 128 * 128

        cfg = LLaDAConfig(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            mlp_hidden_size=mlp_hidden_size,
            activation_type="silu", block_type="llama",
            layer_norm_type="rms", rms_norm_eps=1e-6,
            rope=True, rope_full_precision=True,
            flash_attention=False, weight_tying=False,
            vocab_size=vocab_size, embedding_size=embedding_size,
            max_sequence_length=max_position,
            attention_dropout=dropout, residual_dropout=dropout,
            embedding_dropout=dropout,
            include_bias=False, include_qkv_bias=False,
            scale_logits=False, init_fn="normal", init_std=0.02,
            mask_token_id=mcfg.MASK_ID, pad_token_id=mcfg.PAD_ID,
            mrope_section=None,
        )
        self.model = LLaDAModelLM(cfg, init_params=True)
        self._vocab_size = vocab_size
        self.register_buffer("_bidi_bias", torch.zeros(1, 1, 1, 1),
                             persistent=False)
        log.info(f"LLaDA: vocab={vocab_size}, embed={embedding_size}, "
                 f"d={d_model}, heads={n_heads}, layers={n_layers}, "
                 f"mlp={mlp_hidden_size}")

    def _forward_logits(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_bias=self._bidi_bias,
            use_cache=False,
        ).logits[:, :, :self._vocab_size]

    def _loss(self, logits, targets, mask, t, use_t_weighting):
        if not mask.any():
            return logits[..., 0].float().sum() * 0.0
        selected = logits[mask]
        labels = targets[mask]
        weights = torch.where(labels == mcfg.PAD_ID, self.pad_loss_weight, 1.0).to(selected.dtype)
        loss = F.cross_entropy(selected, labels, reduction='none') * weights
        if use_t_weighting:
            weights_t = (1.0 / t.clamp(min=0.05)).to(selected.dtype)
            loss = loss * weights_t[:, None].expand_as(mask)[mask]
        return loss.mean()

    def forward(self, input_ids, attention_mask, target_ids, loss_mask,
                sc_target_mask, mask_ratio_t, token_type_ids=None,
                sc_loss_weight=1.0, use_t_weighting=True):
        logits = self._forward_logits(input_ids, attention_mask)
        loss = self._loss(logits, target_ids, loss_mask, mask_ratio_t, use_t_weighting)
        if sc_loss_weight:
            corrected = self._forward_logits(logits.detach().argmax(-1), attention_mask)
            loss = loss + sc_loss_weight * self._loss(
                corrected, target_ids, sc_target_mask, mask_ratio_t, use_t_weighting)
        return loss

    @torch.no_grad()
    def get_logits(self, input_ids, attention_mask, token_type_ids=None):
        return self._forward_logits(input_ids, attention_mask)
