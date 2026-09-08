"""HF draft config -> megatron-core TransformerConfig.

只映射 mcore 真正会读的字段。DSpark 特有的东西(mHC、attn_sink、block_size、
markov/confidence head)不进 TransformerConfig —— 它们由自定义模块从 HF config 读。

映射里最要紧的是 MoE 那一组:V4 的路由语义在 mcore 里全有对应字段,所以 router
不用自己写(HF 版的 DeepSeekV4Gate 可以整个丢掉)。逐条对应关系见下面注释。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class DSparkParallelOptions:
    """并行 / 精度选项。默认值 = 单卡 bf16,和现有 FSDP2 路径行为一致。"""

    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    sequence_parallel: bool = False

    # MoE 实现选择
    moe_grouped_gemm: bool = True          # 256 专家必开:否则 256 次小 GEMM
    moe_token_dispatcher_type: str = "alltoall"   # EP>1 时用 alltoall / flex
    moe_enable_deepep: bool = False        # flex + deepep 才生效

    # 精度
    bf16: bool = True
    fp8: Optional[str] = None              # None / "e4m3" / "hybrid"
    fp8_recipe: str = "blockwise"          # 对齐现有 deepspec-v1-fp8-te 分支
    fp8_param: bool = False                # 参数也存 fp8(优化器主权重仍 fp32)
    use_transformer_engine: bool = False   # fp8 必须为 True

    # 显存
    recompute_moe_layer: bool = True       # 等价于现有 gradient_checkpointing

    def __post_init__(self):
        if self.fp8 is not None and not self.use_transformer_engine:
            raise ValueError(
                "fp8 需要 use_transformer_engine=True —— fp8 GEMM 住在 TE 里,"
                "local 后端下 fp8 字段不生效。"
            )
        if self.fp8_param and self.fp8 is None:
            raise ValueError("fp8_param 依赖 fp8:它只改存储,计算由 fp8 开关决定。")
        if self.moe_grouped_gemm and not self.use_transformer_engine:
            # LocalSpecProvider.grouped_mlp_modules 无条件返回 SequentialMLP,
            # 完全忽略 moe_grouped_gemm —— 会静默回退成 256 次小 GEMM,
            # 而且权重布局也跟着变(weight1/weight2 vs local_experts.*)。
            # grouped GEMM 只存在于 TE 后端(TEColumnParallelGroupedLinear)。
            raise ValueError(
                "moe_grouped_gemm=True 需要 use_transformer_engine=True —— "
                "grouped GEMM 只在 TE 后端有实现,local 后端会静默回退成 SequentialMLP。"
                "256 专家不开 grouped GEMM 会退化成 256 次小 GEMM。"
            )


def build_mcore_config(draft_config, opts: DSparkParallelOptions) -> TransformerConfig:
    """draft_config 是 build_flash_draft_config() 产出的 HF config。"""
    hidden = int(draft_config.hidden_size)
    num_heads = int(draft_config.num_attention_heads)
    head_dim = int(draft_config.head_dim)
    n_groups = int(draft_config.o_groups)
    tp = int(opts.tensor_model_parallel_size)

    # TP 切分约束:注意力按 head 切,而 wo_a 是按 group 分块的低秩投影,
    # 两者必须切在同一个边界上,否则 rank 上的 head 和 group 对不齐。
    if num_heads % tp:
        raise ValueError(f"num_attention_heads({num_heads}) 必须被 TP({tp}) 整除")
    if n_groups % tp:
        raise ValueError(
            f"o_groups({n_groups}) 必须被 TP({tp}) 整除 —— wo_a 按 group 切,"
            "group 不整除时 head 和 group 的切分边界对不上。"
        )

    moe_inter = int(draft_config.moe_intermediate_size)
    n_shared = int(getattr(draft_config, "n_shared_experts", 1))
    swiglu_limit = float(getattr(draft_config, "swiglu_limit", 0.0))

    cfg = TransformerConfig(
        # ---- 形状 ----
        num_layers=int(draft_config.num_hidden_layers),
        hidden_size=hidden,
        num_attention_heads=num_heads,
        kv_channels=head_dim,          # mcore 用 kv_channels 表示每头维度
        ffn_hidden_size=moe_inter,     # 全是 MoE 层,这个值实际不被用到
        normalization="RMSNorm",
        layernorm_epsilon=float(draft_config.rms_norm_eps),
        add_bias_linear=bool(getattr(draft_config, "attention_bias", False)),
        gated_linear_unit=True,        # SwiGLU
        activation_func=F.silu,
        # swiglu_limit -> mcore 的 linear_fc1 输出钳位。0 表示不钳。
        activation_func_clamp_value=(swiglu_limit if swiglu_limit > 0 else None),
        hidden_dropout=0.0,
        attention_dropout=float(getattr(draft_config, "attention_dropout", 0.0)),

        # ---- MoE:V4 路由语义逐条对应 ----
        num_moe_experts=int(draft_config.n_routed_experts),
        moe_ffn_hidden_size=moe_inter,
        moe_router_topk=int(draft_config.num_experts_per_tok),
        # scoring_func: "sqrtsoftplus" / "sigmoid" / "softmax" —— mcore 同名支持
        moe_router_score_function=str(draft_config.scoring_func),
        # routed_scaling_factor
        moe_router_topk_scaling_factor=float(draft_config.routed_scaling_factor),
        # noaux_tc:可学的 per-expert 选择偏置(只影响 topk 选择,不进权重)
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=float(
            getattr(draft_config, "router_bias_update_rate", 1e-3)
        ),
        # draft 从头训 MoE 时保留 aux loss 防专家塌缩;热启动预训练权重可调小。
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=float(getattr(draft_config, "moe_aux_loss_coeff", 1e-3)),
        moe_router_dtype="fp32",       # HF 版 gate 也是 .float() 算分
        moe_shared_expert_intermediate_size=(moe_inter * n_shared if n_shared else None),
        moe_grouped_gemm=bool(opts.moe_grouped_gemm),
        moe_token_dispatcher_type=str(opts.moe_token_dispatcher_type),
        moe_enable_deepep=bool(opts.moe_enable_deepep),
        moe_layer_recompute=bool(opts.recompute_moe_layer),
        # fp8 下 MoE 的 ragged token 维要补齐到量化对齐边界。
        # 这是现有 torchao 分支里手工 pad_inner_dim 兜的那件事。
        moe_router_padding_for_quantization=(opts.fp8 is not None),

        # ---- 并行 ----
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=int(opts.expert_model_parallel_size),
        expert_tensor_parallel_size=opts.expert_tensor_parallel_size,
        sequence_parallel=bool(opts.sequence_parallel),

        # ---- 精度 ----
        bf16=bool(opts.bf16),
        params_dtype=torch.bfloat16 if opts.bf16 else torch.float32,
        pipeline_dtype=torch.bfloat16 if opts.bf16 else torch.float32,
        fp8=opts.fp8,
        fp8_recipe=opts.fp8_recipe,
        fp8_param=bool(opts.fp8_param),
    )
    # 自定义模块要从 config 读的 DSpark 专有字段,挂在 TransformerConfig 上带过去,
    # 免得每个模块都再接一个 hf_config 参数。
    cfg.dspark_head_dim = head_dim
    cfg.dspark_rope_dim = int(draft_config.qk_rope_head_dim)
    cfg.dspark_q_lora_rank = int(draft_config.q_lora_rank)
    cfg.dspark_o_lora_rank = int(draft_config.o_lora_rank)
    cfg.dspark_o_groups = n_groups
    cfg.dspark_hc_mult = int(getattr(draft_config, "hc_mult", 4))
    cfg.dspark_hc_sinkhorn_iters = int(getattr(draft_config, "hc_sinkhorn_iters", 20))
    cfg.dspark_hc_eps = float(getattr(draft_config, "hc_eps", 1e-6))
    return cfg


__all__ = ["DSparkParallelOptions", "build_mcore_config"]
