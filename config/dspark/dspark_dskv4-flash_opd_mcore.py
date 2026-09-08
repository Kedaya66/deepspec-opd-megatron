"""OPD + megatron-core 后端(EP / fp8)。

和 dspark_dskv4-flash_opd.py 的差别只在 model.mcore 这一段 —— 数据、loss、
OPD rollout、logging 全部照搬,因为移植只换了 decoder 层。

并行方案选择(4 卡):
  EP=4              专家全切开,通信最省,global batch 靠梯度累积。默认。
  EP=2 + DP=2       折中
  EP=4 + ZeRO-1     专家按 EP 切 + 优化器状态按 DP 切(Adam 状态是显存第二大头)

为什么 EP 而不是继续 ZeRO-3:ZeRO-3 切的是"张量切片",走到 MoE 层要 all-gather
回全部 256 个专家的权重(哪怕每 token 只用 8 个);EP 切的是"专家归属",权重永远
不搬,改成 all-to-all 搬 token。MoE 模型上后者通信量低一到两个数量级。
"""

import os

from deepspec.trainer import DeepSeekV4FlashDSparkOnlineTrainer
from deepspec.utils.constant import BASE_TB_DIR, DSK_V4_FLASH

project_name = "deepspec"
exp_name = "dspark_opd_mcore_ep4"
seed = 42

PRETRAINED_DSPARK = "/shenlb/zwf-spec/train/model/DeepSeek-V4-Flash-DSpark-bf16"

model = dict(
    target_model_name_or_path=DSK_V4_FLASH,
    block_size=5,
    num_draft_layers=3,
    target_layer_ids=[40, 41, 42],

    mask_token_id=128799,
    num_anchors=128,

    ## markov head
    markov_rank=256,
    markov_head_type='vanilla',

    ## confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,

    pretrained_dspark_path=PRETRAINED_DSPARK,
    load_pretrained_weights=True,
    gradient_checkpointing=True,

    ## ---- megatron-core 后端 ----
    # 这一段对应 DSparkParallelOptions;不写这段就走原来的 HF+FSDP2 路径。
    backend="mcore",
    mcore=dict(
        tensor_model_parallel_size=1,      # 3 层 draft,TP 收益小;先不开
        expert_model_parallel_size=4,      # 256 专家 -> 每卡 64
        expert_tensor_parallel_size=1,
        sequence_parallel=False,

        moe_grouped_gemm=True,             # 256 专家必开
        moe_token_dispatcher_type="alltoall",
        moe_enable_deepep=False,           # 先用 alltoall 拿基线,再试 deepep

        bf16=True,
        # fp8 先关。理由:draft 只 3 层,瓶颈是 target 特征导出不是 draft 算力;
        # 要开就照抄 deepspec-v1-fp8-te 的配方(下面两行 + use_transformer_engine)
        fp8=None,                          # "hybrid"
        fp8_recipe="blockwise",            # DeepSeek 式 1x128/128x128 块级缩放
        fp8_param=False,                   # 参数存 fp8;优化器主权重仍 fp32
        use_transformer_engine=False,      # fp8=True 时必须一起打开

        recompute_moe_layer=True,          # 等价于 gradient_checkpointing
    ),
)

train = dict(
    trainer_cls=DeepSeekV4FlashDSparkOnlineTrainer,
    lr=1.0e-5,
    warmup_ratio=0.05,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=2,
    max_train_steps=None,
    max_grad_norm=1.0,
    # EP=4 时不再走 FSDP full_shard —— 专家已按 EP 切,优化器状态用 mcore
    # 的分布式优化器(ZeRO-1)再按 DP 切。
    sharding_strategy=None,
    use_distributed_optimizer=True,
    cpu_offload=False,
    torch_compile=False,               # mcore 的 MoE 路径与 compile 未验证
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=1,
    save_total_limit=2,
)

data = dict(
    target_cache_path=None,
    train_data_path=[
        "/shenlb/zwf-spec/dataset/dataset-v1/主agent输入_医疗助手_正式环境_perfectblend_openai_conv_regen.jsonl",
    ],
    chat_template="v4-flash",
    max_length=45000,
    num_workers=4,
    min_loss_tokens=0,
)

online = dict(
    sglang_server_address=["127.0.0.1:30000", "127.0.0.1:30010"],
    request_timeout=1200.0,
    opd_rollout=True,
    opd_max_new_tokens=512,
    opd_temperature=0.0,
    opd_top_p=1.0,
    opd_prefetch=8,
    opd_max_prefetch=16,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join("/root/checkpoints", project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg
