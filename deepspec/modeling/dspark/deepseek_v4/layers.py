"""DeepSeek-V4 backbone building blocks for DSpark draft training.

These blocks re-implement the trainable core of the DeepSeek-V4 architecture
(Multi-head Latent Attention + Mixture-of-Experts) in dense ``bfloat16`` form so
they can be trained inside the DeepSpec DSpark scaffolding.

The reference inference model
(``deepseek-ai/DeepSeek-V4-{Pro,Flash}-DSpark/inference/model.py``) is a
serving artifact built on custom CUDA kernels (FP8/FP4 GEMMs, ``sparse_attn``
sliding-window attention, KV compression + ``Indexer``, and the
``hc_split_sinkhorn`` Hyper-Connection mixer). Those pieces are inference-time
optimizations whose kernels are not public, and the DSpark draft is trained from
scratch (only ``embed_tokens``/``lm_head`` are copied from the target and
frozen), so they are intentionally not reproduced here. What is reproduced is
the load-bearing, trainable compute that defines V4: the MLA attention shape
(low-rank Q, single latent KV, partial RoPE, grouped low-rank output) and the
MoE FFN (routed experts + shared expert).
"""

from typing import Callable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)


class DeepSeekV4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype))


def _yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    import math

    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    import math

    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001
    ramp = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return ramp.clamp(0, 1)


class DeepSeekV4RotaryEmbedding(nn.Module):
    """Partial-dimension rotary embedding with optional YaRN scaling.

    Only the last ``qk_rope_head_dim`` channels of each head are rotated; the
    remaining channels are position-independent (NoPE), matching MLA.
    """

    def __init__(self, config):
        super().__init__()
        import math

        self.rope_dim = int(config.qk_rope_head_dim)
        base = float(config.rope_theta)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim)
        )
        self.attention_scaling = 1.0

        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling and str(rope_scaling.get("type", rope_scaling.get("rope_type"))) == "yarn":
            factor = float(rope_scaling["factor"])
            orig_max = int(
                rope_scaling.get(
                    "original_max_position_embeddings",
                    getattr(config, "max_position_embeddings", 4096),
                )
            )
            beta_fast = float(rope_scaling.get("beta_fast", 32))
            beta_slow = float(rope_scaling.get("beta_slow", 1))
            low, high = _yarn_correction_range(
                beta_fast, beta_slow, self.rope_dim, base, orig_max
            )
            smooth = 1.0 - _yarn_linear_ramp(low, high, self.rope_dim // 2)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
            # DeepSeek attention temperature scaling for YaRN.
            self.attention_scaling = 0.1 * math.log(factor) + 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        inv_freq = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    unsqueeze_dim: int = 1,
):
    """Apply RoPE to the trailing ``rope_dim`` channels of ``q`` and ``k``.

    ``q`` may be shorter than ``k`` along the sequence dim (draft block query vs
    context+block keys); its rotation uses the trailing ``q_len`` positions.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)

    q_pass, q_rot = q[..., :-rope_dim], q[..., -rope_dim:]
    k_pass, k_rot = k[..., :-rope_dim], k[..., -rope_dim:]

    q_rot = (q_rot * cos[..., -q_len:, :]) + (rotate_half(q_rot) * sin[..., -q_len:, :])
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q = torch.cat([q_pass, q_rot], dim=-1)
    k = torch.cat([k_pass, k_rot], dim=-1)
    return q, k


class DeepSeekV4DSparkAttention(nn.Module):
    """Multi-head Latent Attention adapted to DSpark bidirectional block attention.

    Mirrors ``Qwen3DSparkAttention``: keys/values are the concatenation of the
    projected target context (``target_hidden_states``) and the draft/noise
    block (``hidden_states``). MLA specifics: low-rank query projection
    (``wq_a`` -> ``q_norm`` -> ``wq_b``), a single low-rank latent KV head shared
    across query heads (MQA), partial RoPE, and a grouped low-rank output
    projection (``wo_a`` -> ``wo_b``).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.rope_dim = int(config.qk_rope_head_dim)
        self.q_lora_rank = int(config.q_lora_rank)
        self.o_lora_rank = int(config.o_lora_rank)
        self.n_groups = int(config.o_groups)
        assert self.num_heads % self.n_groups == 0, (
            "num_attention_heads must be divisible by o_groups."
        )
        self.eps = float(config.rms_norm_eps)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        self.is_causal = False
        # The single latent KV head is expanded to ``num_heads`` in ``forward``
        # (plain MHA), so the attention kernels must not repeat it again.
        self.num_key_value_heads = self.num_heads
        self.num_key_value_groups = 1

        attn_bias = bool(getattr(config, "attention_bias", False))
        # Query: low-rank down-projection then per-head up-projection.
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=attn_bias)
        self.q_norm = DeepSeekV4RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        # Key/Value: single latent head of size head_dim.
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=attn_bias)
        self.kv_norm = DeepSeekV4RMSNorm(self.head_dim, eps=self.eps)
        # Output: grouped low-rank projection.
        self.wo_a = nn.Linear(
            self.num_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.hidden_size, bias=attn_bias)

    def _project_latent_kv(self, x: torch.Tensor) -> torch.Tensor:
        # [b, seq, head_dim] single latent head.
        return self.kv_norm(self.wkv(x))

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]

        q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        # Weight-free per-head RMS on the query (matches the reference MLA).
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q = q.transpose(1, 2)

        kv_ctx = self._project_latent_kv(target_hidden_states)
        kv_noise = self._project_latent_kv(hidden_states)
        kv = torch.cat([kv_ctx, kv_noise], dim=1)
        # Single latent head -> broadcast to num_heads (MQA).
        kv = kv.view(bsz, ctx_len + q_len, 1, self.head_dim).transpose(1, 2)
        # Broadcast the single latent head to all query heads (MQA).
        k = kv.expand(bsz, self.num_heads, ctx_len + q_len, self.head_dim)
        # Value is the unrotated latent; RoPE below only rewrites ``k``.
        v = kv.expand(bsz, self.num_heads, ctx_len + q_len, self.head_dim)

        cos, sin = position_embeddings
        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin, self.rope_dim)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_is_causal = bool(kwargs.get("is_causal", False))
        self.is_causal = attn_is_causal
        kwargs["is_causal"] = attn_is_causal
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a)
        return self.wo_b(attn_output.flatten(2)), attn_weights


class DeepSeekV4Expert(nn.Module):
    """SwiGLU expert with an optional activation clamp (``swiglu_limit``)."""

    def __init__(self, hidden_size: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inter_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, hidden_size, bias=False)
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class DeepSeekV4Gate(nn.Module):
    """Routed-expert gate with sigmoid / softmax / sqrt-softplus scoring.

    Uses the ``noaux_tc`` selection scheme: a learnable per-expert bias shifts
    scores for top-k *selection* only; routing *weights* come from the unbiased
    scores. Hash routing from the target model is dropped (the draft has no
    token->expert table), so every draft MoE layer routes by score.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = int(config.num_experts_per_tok)
        self.n_routed_experts = int(config.n_routed_experts)
        self.score_func = str(config.scoring_func)
        self.route_scale = float(config.routed_scaling_factor)
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        # Router projection (kept as a Linear so PreTrainedModel weight init
        # covers it). ``bias`` is the noaux_tc selection bias, trained from zero.
        self.router = nn.Linear(int(config.hidden_size), self.n_routed_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.n_routed_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.router.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # "sqrtsoftplus"
            scores = F.softplus(scores).sqrt()
        indices = (scores + self.bias).topk(self.top_k, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.score_func != "softmax" and self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights * self.route_scale
        return weights, indices


class DeepSeekV4MoE(nn.Module):
    """Mixture-of-Experts FFN: top-k routed experts plus one shared expert."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.n_routed_experts = int(config.n_routed_experts)
        inter_dim = int(config.moe_intermediate_size)
        swiglu_limit = float(getattr(config, "swiglu_limit", 0.0))
        self.gate = DeepSeekV4Gate(config)
        self.experts = nn.ModuleList(
            [
                DeepSeekV4Expert(self.hidden_size, inter_dim, swiglu_limit)
                for _ in range(self.n_routed_experts)
            ]
        )
        n_shared = int(getattr(config, "n_shared_experts", 1))
        assert n_shared == 1, "DeepSeek-V4 DSpark draft supports a single shared expert."
        self.shared_expert = DeepSeekV4Expert(
            self.hidden_size, inter_dim, swiglu_limit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        weights, indices = self.gate(x)
        out = torch.zeros_like(x)
        for expert_id in range(self.n_routed_experts):
            token_idx, slot_idx = torch.where(indices == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_out = self.experts[expert_id](x[token_idx])
            expert_out = expert_out * weights[token_idx, slot_idx].unsqueeze(-1).to(
                expert_out.dtype
            )
            out.index_add_(0, token_idx, expert_out)
        # FSDP reduce-scatters gradients across ranks and needs every expert's params
        # to receive a gradient on every rank; an expert that routes 0 tokens on some
        # rank would otherwise produce an inconsistent (empty) grad and crash backward.
        # Add a mathematically-zero term that keeps all expert params in the graph.
        if self.training:
            touch = x.new_zeros(())
            for expert in self.experts:
                touch = touch + (
                    expert.gate_proj.weight.sum()
                    + expert.up_proj.weight.sum()
                    + expert.down_proj.weight.sum()
                )
            out = out + touch * 0.0
        out = out + self.shared_expert(x)
        return out.view(shape)


class DeepSeekV4DSparkDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = DeepSeekV4DSparkAttention(config=config, layer_idx=layer_idx)
        self.mlp = DeepSeekV4MoE(config)
        self.input_layernorm = DeepSeekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepSeekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids, output_attentions, use_cache
        assert hidden_states is not None
        assert target_hidden_states is not None
        assert position_embeddings is not None

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


__all__ = [
    "DeepSeekV4RMSNorm",
    "DeepSeekV4RotaryEmbedding",
    "DeepSeekV4DSparkAttention",
    "DeepSeekV4Expert",
    "DeepSeekV4Gate",
    "DeepSeekV4MoE",
    "DeepSeekV4DSparkDecoderLayer",
]
