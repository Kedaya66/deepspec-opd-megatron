"""HF match-v1 权重 -> megatron-core 权重(按 TP / EP rank 切片)。

只有两处布局真的变了,其余都是同名搬家:
  1) MoE 专家:HF 是 per-expert 的 gate_proj/up_proj/down_proj,
     mcore 是 linear_fc1(= cat[gate, up]) / linear_fc2(= down)
  2) 分组低秩输出:HF 的 wo_a 是 nn.Linear(展平存),mcore 存成 3 维 Parameter

切片规则(和 mcore_layers.py 里的 TP 设计一一对应):
  wq_b       按输出行切(head)          [tp_rank]
  wo_a       按 group 切(dim 0)        [tp_rank]
  wo_b       按输入列切(group*o_lora)   [tp_rank]
  attn_sink  按 head 切                [tp_rank]
  experts    按专家切                   [ep_rank]
  其余(低秩下投影 / norm / mHC / router) 全复制,不切。

注意力段和 MoE 段都是**按需转换**:sd 里没有对应键就跳过。这样可以只转注意力
或只转 MoE(测试要用),也支持增量转换。
"""

from typing import Dict

import torch


def _slice(t: torch.Tensor, dim: int, rank: int, world: int) -> torch.Tensor:
    if world == 1:
        return t
    n = t.shape[dim]
    assert n % world == 0, f"dim {dim} 大小 {n} 不能被 {world} 整除"
    step = n // world
    return t.narrow(dim, rank * step, step).contiguous()


def convert_layer_state_dict(
    hf_sd: Dict[str, torch.Tensor],
    *,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    o_lora_rank: int,
    num_experts: int,
    tp_rank: int = 0,
    tp_size: int = 1,
    ep_rank: int = 0,
    ep_size: int = 1,
    moe_grouped_gemm: bool = False,
    expert_tensor_parallel_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """hf_sd 是**单层**的 state_dict(键不带 layers.{i}. 前缀)。"""
    assert expert_tensor_parallel_size == 1, (
        "expert_tensor_parallel_size>1 需要再对专家权重切一刀,当前未实现"
    )
    out: Dict[str, torch.Tensor] = {}

    # ================= 注意力 =================
    if "self_attn.wq_b.weight" in hf_sd:
        # ---- 复制的部分:低秩下投影 + norm ----
        for k in ("wq_a.weight", "wq_a.bias", "wkv.weight", "wkv.bias",
                  "q_norm.weight", "kv_norm.weight", "wo_b.bias"):
            src = f"self_attn.{k}"
            if src in hf_sd:
                out[f"self_attn.{k}"] = hf_sd[src].clone()

        # ---- 按 head 切 ----
        out["self_attn.wq_b.weight"] = _slice(
            hf_sd["self_attn.wq_b.weight"], 0, tp_rank, tp_size)
        out["self_attn.attn_sink"] = _slice(
            hf_sd["self_attn.attn_sink"], 0, tp_rank, tp_size)

        # ---- 分组低秩输出 ----
        # HF: nn.Linear(weight=[n_groups*o_lora, heads_per_group*head_dim]),
        #     forward 里 .view(n_groups, o_lora, -1) 再 einsum
        wo_a = hf_sd["self_attn.wo_a.weight"].view(n_groups, o_lora_rank, -1)
        out["self_attn.wo_a"] = _slice(wo_a, 0, tp_rank, tp_size)
        # wo_b: RowParallelLinear 按输入维切,输入维就是 n_groups*o_lora
        out["self_attn.wo_b.weight"] = _slice(
            hf_sd["self_attn.wo_b.weight"], 1, tp_rank, tp_size)

    # ================= norm / mHC:全复制 =================
    for k in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        if k in hf_sd:
            out[k] = hf_sd[k].clone()
    for hc in ("attn_hc", "ffn_hc"):
        for p in ("fn", "base", "scale"):
            src = f"{hc}.{p}"
            if src in hf_sd:
                out[src] = hf_sd[src].clone()

    # ================= MoE =================
    if not num_experts or "mlp.gate.router.weight" not in hf_sd:
        return out

    # ---- router:HF 的 gate.router 是 nn.Linear;gate.bias 是 noaux_tc 选择偏置 ----
    out["mlp.router.weight"] = hf_sd["mlp.gate.router.weight"].clone()
    if "mlp.gate.bias" in hf_sd:
        out["mlp.router.expert_bias"] = hf_sd["mlp.gate.bias"].clone()

    # ---- 专家:布局要变 ----
    assert num_experts % ep_size == 0, "专家数必须被 EP 整除"
    n_local = num_experts // ep_size
    first = ep_rank * n_local

    # 两种 experts 实现的参数命名(实测 megatron-core 0.19 / TE 2.16):
    #   SequentialMLP : experts.local_experts.{le}.linear_fc{1,2}.weight
    #   TEGroupedMLP  : experts.linear_fc{1,2}.weight{le}
    # **形状完全一样**,只是命名不同 —— TEGroupedMLP 不是把专家融合成一个大张量,
    # 它仍然每个本地专家一个 Parameter,融合发生在 kernel 层。
    # (legacy GroupedMLP 才是 weight1/weight2 那种拼接布局,这里用不到。)
    for le in range(n_local):
        ge = first + le                      # 全局专家号
        gate = hf_sd[f"mlp.experts.{ge}.gate_proj.weight"]
        up = hf_sd[f"mlp.experts.{ge}.up_proj.weight"]
        down = hf_sd[f"mlp.experts.{ge}.down_proj.weight"]
        # mcore 的 gated linear unit:linear_fc1 输出是 [gate; up] 拼接
        fc1 = torch.cat([gate, up], dim=0).contiguous()
        if moe_grouped_gemm:
            out[f"mlp.experts.linear_fc1.weight{le}"] = fc1
            out[f"mlp.experts.linear_fc2.weight{le}"] = down.clone()
        else:
            out[f"mlp.experts.local_experts.{le}.linear_fc1.weight"] = fc1
            out[f"mlp.experts.local_experts.{le}.linear_fc2.weight"] = down.clone()

    # ---- 共享专家 ----
    if "mlp.shared_expert.gate_proj.weight" in hf_sd:
        sg = hf_sd["mlp.shared_expert.gate_proj.weight"]
        su = hf_sd["mlp.shared_expert.up_proj.weight"]
        sd_ = hf_sd["mlp.shared_expert.down_proj.weight"]
        out["mlp.shared_experts.linear_fc1.weight"] = torch.cat([sg, su], dim=0)
        out["mlp.shared_experts.linear_fc2.weight"] = sd_.clone()

    return out


def convert_model_state_dict(hf_sd, *, num_layers, **kw):
    """整模型版本:逐层转换 + 顶层参数(embedding/heads/mHC head)原样搬。"""
    out = {}
    for i in range(num_layers):
        prefix = f"layers.{i}."
        layer_sd = {k[len(prefix):]: v for k, v in hf_sd.items() if k.startswith(prefix)}
        for k, v in convert_layer_state_dict(layer_sd, **kw).items():
            out[prefix + k] = v
    # 层以外的东西 mcore 版没改结构,直接搬
    for k, v in hf_sd.items():
        if not k.startswith("layers."):
            out[k] = v.clone()
    return out


__all__ = ["convert_layer_state_dict", "convert_model_state_dict"]
