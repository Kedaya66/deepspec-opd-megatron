"""DeepSeek-V4 DSpark draft layers, match-v1: numerically faithful to the official
``inference/model.py``.

Differences vs ``layers.py`` (each verified against real checkpoint weights in
``.tmp_arch_check/``):

* **V-RoPE** — the shared KV latent's rope slice is rotated once (so V carries
  RoPE, exactly like the reference where K and V are the same tensor), and the
  attention output's rope slice is inverse-rotated at query positions. The KV
  entry's contribution then depends only on its relative distance to the query.
* **attn_sink** — per-head learnable sink logit joins the softmax normalizer
  (gpt-oss style). Requires the inline eager attention below; SDPA cannot
  express it.
* **Hyper-Connections (mHC)** — the residual is ``hc_mult`` parallel streams
  ``[B, S, hc, D]``; each sublayer reads via a learned collapse (``pre``) and
  writes back via ``post`` plus a Sinkhorn-projected ``comb`` mixing matrix.
  Semantics copied from the official trainable implementation
  (``transformers.models.deepseek_v4.DeepseekV4HyperConnection``), which is
  algebraically identical to the reference's ``hc_split_sinkhorn`` kernel.

The rope-channel permutation (interleaved -> rotate_half layout) is a
load-time weight transform: see ``load_pretrained_match_v1.py``. With permuted
weights, the rotate_half convention used here is bit-equivalent to the
reference's interleaved convention on original weights.
"""

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from transformers.modeling_layers import GradientCheckpointingLayer

from deepspec.modeling.dspark.deepseek_v4.layers import (
    DeepSeekV4DSparkAttention,
    DeepSeekV4MoE,
    DeepSeekV4RMSNorm,
    rotate_half,
)


def apply_rope_trailing(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                        rope_dim: int, inverse: bool = False) -> torch.Tensor:
    """Rotate the trailing ``rope_dim`` channels of ``x`` ([B, H, S, D])."""
    x_pass, x_rot = x[..., :-rope_dim], x[..., -rope_dim:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    if inverse:
        sin = -sin
    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([x_pass, x_rot], dim=-1)


class DeepSeekV4DSparkAttentionMatchV1(DeepSeekV4DSparkAttention):
    """MLA DSpark attention with V-RoPE + output de-rotation + attention sink."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # fp32 like the checkpoint; zero init only matters for from-scratch runs.
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        cos, sin = position_embeddings

        q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q = q.transpose(1, 2)
        q = apply_rope_trailing(q, cos[:, -q_len:], sin[:, -q_len:], self.rope_dim)

        kv_ctx = self._project_latent_kv(target_hidden_states)
        kv_noise = self._project_latent_kv(hidden_states)
        kv = torch.cat([kv_ctx, kv_noise], dim=1)
        kv = kv.view(bsz, ctx_len + q_len, 1, self.head_dim).transpose(1, 2)
        # Rotate the single latent once: it serves as BOTH key and value, exactly
        # like the reference where apply_rotary_emb writes into kv in place.
        kv = apply_rope_trailing(kv, cos, sin, self.rope_dim)
        k = kv.expand(bsz, self.num_heads, ctx_len + q_len, self.head_dim)
        v = k

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + attention_mask
        sinks = self.attn_sink.to(scores.dtype).view(1, -1, 1, 1).expand(bsz, -1, q_len, 1)
        combined = torch.cat([scores, sinks], dim=-1)
        combined = combined - combined.amax(dim=-1, keepdim=True)
        probs = torch.softmax(combined.float(), dim=-1).to(v.dtype)
        attn = probs[..., :-1]
        if self.training and self.attention_dropout > 0:
            attn = F.dropout(attn, p=self.attention_dropout)

        o = torch.matmul(attn, v)
        # De-rotate at query positions: KV contributions become relative-position.
        o = apply_rope_trailing(o, cos[:, -q_len:], sin[:, -q_len:], self.rope_dim, inverse=True)

        o = o.transpose(1, 2).reshape(bsz, q_len, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2)), None


class DSparkHyperConnection(nn.Module):
    """mHC collapse/expand weights; semantics identical to
    ``transformers.models.deepseek_v4.DeepseekV4HyperConnection``."""

    def __init__(self, hidden_size: int, hc_mult: int = 4, sinkhorn_iters: int = 20,
                 hc_eps: float = 1e-6, rms_eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.rms_eps = rms_eps
        mix = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(torch.zeros(mix, hc_mult * hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(mix, dtype=torch.float32))
        self.scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, hidden_streams: torch.Tensor):
        # hidden_streams: [B, S, hc, D]
        hc = self.hc_mult
        flat = hidden_streams.flatten(start_dim=2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        mixes = F.linear(flat, self.fn.float())
        pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class DSparkHyperHead(nn.Module):
    """Final 4->1 stream collapse before the shared RMSNorm / lm_head."""

    def __init__(self, hidden_size: int, hc_mult: int = 4, hc_eps: float = 1e-6,
                 rms_eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.rms_eps = rms_eps
        self.fn = nn.Parameter(torch.zeros(hc_mult, hc_mult * hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(hc_mult, dtype=torch.float32))
        self.scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        flat = hidden_streams.flatten(start_dim=2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        mixes = F.linear(flat, self.fn.float())
        pre = torch.sigmoid(mixes * self.scale.float() + self.base.float()) + self.hc_eps
        return (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)


class DeepSeekV4DSparkDecoderLayerMatchV1(GradientCheckpointingLayer):
    """Decoder block with mHC residual streams, mirroring the reference Block."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        hc_mult = int(getattr(config, "hc_mult", 4))
        sinkhorn_iters = int(getattr(config, "hc_sinkhorn_iters", 20))
        hc_eps = float(getattr(config, "hc_eps", 1e-6))
        self.self_attn = DeepSeekV4DSparkAttentionMatchV1(config=config, layer_idx=layer_idx)
        self.mlp = DeepSeekV4MoE(config)
        self.input_layernorm = DeepSeekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepSeekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DSparkHyperConnection(config.hidden_size, hc_mult, sinkhorn_iters,
                                             hc_eps, config.rms_norm_eps)
        self.ffn_hc = DSparkHyperConnection(config.hidden_size, hc_mult, sinkhorn_iters,
                                            hc_eps, config.rms_norm_eps)

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids
        assert hidden_states is not None and hidden_states.ndim == 4  # [B, S, hc, D]
        assert target_hidden_states is not None and position_embeddings is not None
        dtype = hidden_states.dtype

        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(collapsed),
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        # comb consumed transposed: out[j] = post[j]*sublayer + sum_i comb[i, j]*stream_i
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_out = self.mlp(self.post_attention_layernorm(collapsed))
        return post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )


__all__ = [
    "apply_rope_trailing",
    "DeepSeekV4DSparkAttentionMatchV1",
    "DSparkHyperConnection",
    "DSparkHyperHead",
    "DeepSeekV4DSparkDecoderLayerMatchV1",
]
