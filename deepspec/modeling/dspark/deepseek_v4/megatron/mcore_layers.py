"""DSpark match-v1 层的 megatron-core 实现。

和 HF 版(layers_match_v1.py)的分工:
  * 大矩阵 + 跨卡通信 -> mcore 积木(ColumnParallelLinear / RowParallelLinear / MoELayer)
  * 逐元素的怪逻辑   -> 照抄 HF 版(rope 尾旋、attn_sink、mHC+Sinkhorn)

TP 切分的关键约束(之前 TP=2 崩掉的根因):
  * q_lora_rank / head_dim 这两个低秩瓶颈**不能切** —— 它们后面紧跟 RMSNorm,
    norm 需要完整维度;而且只有几百维,切了通信开销大于收益。
  * 能切的只有 head 维:wq_b 按 head 切(ColumnParallelLinear),
    attn_sink 跟着 head 切,wo_a 按 group 切(要求 o_groups % tp == 0,
    这样 rank 上的 head 和 group 边界对齐),wo_b 用 RowParallelLinear 收尾。
  * wkv 产出的单个 latent 被所有 head 共享 -> 必须复制,不能切。
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core import parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    TransformerLayerSubmodules,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """和 HF 版 layers.rotate_half 一致(rotate_half 约定,权重已在加载时置换)。"""
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope_trailing(x, cos, sin, rope_dim: int, inverse: bool = False):
    """旋转 x([B,H,S,D]) 的最后 rope_dim 个通道。逐行对齐 HF 版。"""
    x_pass, x_rot = x[..., :-rope_dim], x[..., -rope_dim:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    if inverse:
        sin = -sin
    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([x_pass, x_rot], dim=-1)


class McoreDSparkAttention(MegatronModule):
    """MLA + V-RoPE + 输出反旋转 + per-head attn_sink,linear 走 mcore 的 TP 版本。"""

    def __init__(self, config, **kw):
        super().__init__(config=config)
        d = config.hidden_size
        self.head_dim = int(config.dspark_head_dim)
        self.rope_dim = int(config.dspark_rope_dim)
        self.q_lora_rank = int(config.dspark_q_lora_rank)
        self.o_lora_rank = int(config.dspark_o_lora_rank)
        self.n_groups_global = int(config.dspark_o_groups)
        self.num_heads_global = int(config.num_attention_heads)
        self.eps = float(config.layernorm_epsilon)
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = float(config.attention_dropout)

        tp = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_size = tp
        # 本 rank 持有的 head / group 数。build_mcore_config 已断言整除。
        self.num_heads = self.num_heads_global // tp
        self.n_groups = self.n_groups_global // tp
        self.heads_per_group = self.num_heads_global // self.n_groups_global
        bias = bool(config.add_bias_linear)

        # --- Q:低秩下投影(复制) -> norm -> 按 head 切的上投影 ---
        # wq_a 复制:输出 q_lora_rank 紧跟 RMSNorm,切了 norm 就错。
        # 复制的 linear 用普通 nn.Linear,但 dtype 必须跟着 config —— 默认 fp32
        # 会和 bf16 激活撞 "mat1 and mat2 have same dtype"。
        self.wq_a = nn.Linear(d, self.q_lora_rank, bias=bias,
                              dtype=config.params_dtype)
        self.q_norm = WrappedTorchNorm(hidden_size=self.q_lora_rank, config=config)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank, self.num_heads_global * self.head_dim, config=config,
            init_method=config.init_method, bias=False,
            gather_output=False, skip_bias_add=True,
        )

        # --- KV:单个 latent,所有 head 共享 -> 必须复制 ---
        self.wkv = nn.Linear(d, self.head_dim, bias=bias,
                             dtype=config.params_dtype)
        self.kv_norm = WrappedTorchNorm(hidden_size=self.head_dim, config=config)

        # --- 输出:分组低秩。wo_a 按 group 切,wo_b RowParallel 收尾 ---
        self.wo_a = nn.Parameter(torch.empty(
            self.n_groups, self.o_lora_rank, self.heads_per_group * self.head_dim,
            dtype=config.params_dtype,
        ))
        # 标记为 TP 切分参数:mcore 的梯度同步会跳过它(不做 TP all-reduce)
        setattr(self.wo_a, "tensor_model_parallel", True)
        setattr(self.wo_a, "partition_dim", 0)
        self.wo_b = RowParallelLinear(
            self.n_groups_global * self.o_lora_rank, d, config=config,
            init_method=config.output_layer_init_method, bias=bias,
            input_is_parallel=True, skip_bias_add=True,
        )

        # per-head 可学 sink,跟着 head 切。fp32 与 HF 版一致。
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        setattr(self.attn_sink, "tensor_model_parallel", True)
        setattr(self.attn_sink, "partition_dim", 0)

        with torch.no_grad():
            self.wo_a.normal_(0.0, float(config.init_method_std))

    def _latent_kv(self, x):
        return self.kv_norm(self.wkv(x))

    def forward(self, hidden_states, target_hidden_states, cos, sin, attention_mask=None):
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden_states.shape[1]
        H = self.num_heads

        q, _ = self.wq_b(self.q_norm(self.wq_a(hidden_states)))
        q = q.view(bsz, q_len, H, self.head_dim)
        # 无权重的 per-head RMS(和参考 MLA 一致)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q = q.transpose(1, 2)
        q = apply_rope_trailing(q, cos[:, -q_len:], sin[:, -q_len:], self.rope_dim)

        kv = torch.cat([self._latent_kv(target_hidden_states),
                        self._latent_kv(hidden_states)], dim=1)
        kv = kv.view(bsz, ctx_len + q_len, 1, self.head_dim).transpose(1, 2)
        # 单 latent 只旋一次:它同时当 K 和 V
        kv = apply_rope_trailing(kv, cos, sin, self.rope_dim)
        k = kv.expand(bsz, H, ctx_len + q_len, self.head_dim)
        v = k

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask,
                                            torch.finfo(scores.dtype).min)
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
        o = apply_rope_trailing(o, cos[:, -q_len:], sin[:, -q_len:],
                                self.rope_dim, inverse=True)
        o = o.transpose(1, 2).reshape(bsz, q_len, self.n_groups, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, self.wo_a)
        # wo_b 是 RowParallelLinear:本 rank 只贡献自己那几个 group 的部分和,
        # 它内部的 all-reduce 把各 rank 的部分和加起来 —— 正好等于全 group 求和。
        out, _ = self.wo_b(o.flatten(2))
        return out


class McoreDSparkHyperConnection(nn.Module):
    """mHC:读(collapse)+写(post/comb)权重。含 Sinkhorn 投影,逐行对齐 HF 版。

    全程 fp32、参数复制(不切 TP)—— 残差流的 D 维没被切,而且这些矩阵很小。
    """

    def __init__(self, hidden_size, hc_mult=4, sinkhorn_iters=20,
                 hc_eps=1e-6, rms_eps=1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.rms_eps = rms_eps
        mix = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(torch.zeros(mix, hc_mult * hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(mix, dtype=torch.float32))
        self.scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, hidden_streams):
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
        # Sinkhorn:交替按行/列归一化,把 comb 投影成近似双随机矩阵
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class McoreDSparkHyperHead(nn.Module):
    """最后一次 4->1 塌缩(进 final norm / lm_head 之前)。"""

    def __init__(self, hidden_size, hc_mult=4, hc_eps=1e-6, rms_eps=1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.rms_eps = rms_eps
        self.fn = nn.Parameter(torch.zeros(hc_mult, hc_mult * hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(hc_mult, dtype=torch.float32))
        self.scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, hidden_streams):
        flat = hidden_streams.flatten(start_dim=2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        mixes = F.linear(flat, self.fn.float())
        pre = torch.sigmoid(mixes * self.scale.float() + self.base.float()) + self.hc_eps
        return (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)


class McoreDSparkLayer(MegatronModule, BaseTransformerLayer):
    """一层 DSpark。mlp 槽由 spec 填(mcore MoELayer),其余自己写。

    forward 返回二元组 (streams, None) —— 这是 TransformerBlock 的契约,
    留着以后想接回 block 拿重算/PP 时不用改。
    """

    def __init__(self, config, submodules: TransformerLayerSubmodules = None,
                 layer_number: int = 1, pg_collection=None, vp_stage=None, **kw):
        super().__init__(config=config)
        self.layer_number = layer_number
        d = config.hidden_size
        hc = int(config.dspark_hc_mult)
        self.self_attn = McoreDSparkAttention(config)
        # mlp 槽的契约是 MlpBuilder 协议(直接调用),不是 ModuleSpec(走 build_module)
        self.mlp = submodules.mlp(config=config, pg_collection=pg_collection,
                                  is_mtp_layer=False, name=None)
        self.input_layernorm = WrappedTorchNorm(hidden_size=d, config=config)
        self.post_attention_layernorm = WrappedTorchNorm(hidden_size=d, config=config)
        self.attn_hc = McoreDSparkHyperConnection(
            d, hc, int(config.dspark_hc_sinkhorn_iters),
            float(config.dspark_hc_eps), float(config.layernorm_epsilon))
        self.ffn_hc = McoreDSparkHyperConnection(
            d, hc, int(config.dspark_hc_sinkhorn_iters),
            float(config.dspark_hc_eps), float(config.layernorm_epsilon))

    def forward(self, hidden_states, target_hidden_states=None,
                cos=None, sin=None, attention_mask=None):
        assert hidden_states.ndim == 4, "隐状态是 4 路残差流 [B,S,hc,D]"
        dtype = hidden_states.dtype

        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_out = self.self_attn(self.input_layernorm(collapsed),
                                  target_hidden_states, cos, sin, attention_mask)
        # 写回:post 广播 + comb 转置混合(替代普通残差 x + f(x))
        hidden_states = (post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2)
                         + torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_states))

        post, comb, collapsed = self.ffn_hc(hidden_states)
        x = self.post_attention_layernorm(collapsed)
        # mcore 的 MoELayer / MLP 一律 sequence-first [s,b,h]。
        # .contiguous() 不能省:MoE router 的 gating 里有 inp.view(-1, h),
        # transpose 出来的非连续张量会直接报 "view size is not compatible"。
        mlp_out, _ = self.mlp(x.transpose(0, 1).contiguous())
        mlp_out = mlp_out.transpose(0, 1).contiguous()
        out = (post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2)
               + torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_states))
        return out, None


__all__ = [
    "rotate_half",
    "apply_rope_trailing",
    "McoreDSparkAttention",
    "McoreDSparkHyperConnection",
    "McoreDSparkHyperHead",
    "McoreDSparkLayer",
]
