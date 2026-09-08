"""OPD + megatron-core 模型(EP)+ megatron trainer(DDP/ZeRO-1)+ 特征预取。

这是把三条分支合到一起的配置:
  * model.backend="mcore"  —— decoder 层用 mcore 实现,256 专家按 EP 切(train-v4 原有)
  * train.trainer_cls=Megatron*  —— mcore DDP + 分布式优化器(从 train-v1/deepspec-megatron 搬来)
  * online.feature_prefetch / opd_rollout —— 并发预取(前者 2026-08-10 实测 70.6→37.5 min)

并行度写了两遍(model.mcore 和 train.megatron),必须一致 ——
megatron/dist.py 里有断言。原因见那里的注释:模型构造中的 init_dspark_parallel
对"已建组"直接 return,不一致会让其中一份被静默忽略。

分阶段验证(每次失败的归因才清晰;加载权重就要 8 分钟,别一次改多个变量):
  阶段 1(当前):随机初始化 + teacher-forced 预取 + EP=4
      验并行组 / mcore 层 / EP 分发 / DDP / 优化器 / 预取
  阶段 2:opd_rollout=True(把下面 online 段的开关调过来)
  阶段 3:load_pretrained_weights=True —— 需要先接 convert_model_state_dict,
      HF layout 的预训练权重不能直接灌进 mcore 模型
"""

import os

from deepspec.trainer import MegatronDeepSeekV4FlashDSparkOnlineTrainer
from deepspec.utils.constant import BASE_TB_DIR, DSK_V4_FLASH

project_name = "deepspec"
exp_name="V24_opd_2ep"
seed = 42

PRETRAINED_DSPARK = "/shenlb/zwf-spec/train/model/DeepSeek-V4-Flash-DSpark-bf16"

# 并行度的单一来源:下面 model.mcore 和 train.megatron 都引用它
EP_SIZE = 4
TP_SIZE = 1

model = dict(
    target_model_name_or_path=DSK_V4_FLASH,
    block_size=5,
    num_draft_layers=3,
    target_layer_ids=[40, 41, 42],

    mask_token_id=128799,
    num_anchors=128,

    markov_rank=256,
    markov_head_type='vanilla',

    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,

    pretrained_dspark_path=PRETRAINED_DSPARK,
    # 阶段 3 已接:走 build_match_v1_state_dict -> convert_model_state_dict,
    # 按 ep_rank 只取本 rank 的专家。转换做过数值对拍(单层最大相对差 1.8e-4)
    # 与 EP 切片检查,加载后还有"无遗漏参数"的断言。
    load_pretrained_weights=True,
    gradient_checkpointing=True,

    ## ---- megatron-core 后端 ----
    backend="mcore",
    mcore=dict(
        tensor_model_parallel_size=TP_SIZE,   # 3 层 draft,TP 收益小
        expert_model_parallel_size=EP_SIZE,   # 256 专家 -> 每卡 64
        expert_tensor_parallel_size=1,
        sequence_parallel=False,

        # 256 专家必须 grouped GEMM,而它只在 TE 后端有实现 ——
        # LocalSpecProvider.grouped_mlp_modules 无条件返回 SequentialMLP
        # (会静默退化成 256 次小 GEMM)。mcore_config.__post_init__ 对
        # grouped_gemm=True + use_transformer_engine=False 直接报错,
        # v4 原有的 dspark_dskv4-flash_opd_mcore.py 正是这个组合,起不来。
        moe_grouped_gemm=True,
        use_transformer_engine=True,
        moe_token_dispatcher_type="alltoall",
        moe_enable_deepep=False,

        bf16=True,
        # v5 默认 hybrid(2026-08-12 实测:-8% 时间,-3GB 显存,loss 差 1e-4)
        fp8="hybrid",
        fp8_recipe="blockwise",
        fp8_param=False,

        # v5 默认关(2026-08-12 实测 gbs=64 warm:计算侧时间 -48%,峰值显存反而 -7GB)
        recompute_moe_layer=False,
    ),
)

train = dict(
    trainer_cls=MegatronDeepSeekV4FlashDSparkOnlineTrainer,
    lr=1.0e-5,
    warmup_ratio=0.05,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=1024,
    num_train_epochs=2,
    max_train_steps=None,
    max_grad_norm=1.0,
    torch_compile=False,          # mcore MoE 路径与 compile 未验证

    megatron=dict(
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=EP_SIZE,
        use_distributed_optimizer=True,
        # 19.85B draft 的 fp32 梯度 buffer 要 73.93 GiB,140GiB 卡上必 OOM(实测)
        grad_reduce_in_fp32=False,
        # MoE draft 必须关:长期收不到 token 的 expert 会触发 mcore 首迭代的
        # bucket golden-count 断言
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        check_for_nan_in_grad=False,
        average_in_collective=True,
        bucket_size=None,
        # 精度感知优化器:12 B/param -> 6 B/param。不开的话 EP=4 下虽然专家已切开,
        # 但非专家参数的优化器状态仍可能吃紧(实测 EP=1 时不开必 OOM)
        use_precision_aware_optimizer=True,
        exp_avg_dtype="bf16",
        exp_avg_sq_dtype="bf16",
        optimizer_cpu_offload=False,
        optimizer_offload_fraction=1.0,
        overlap_cpu_optimizer_d2h_h2d=False,
        use_torch_optimizer_for_cpu_offload=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        min_lr=0.0,
        lr_decay_style="cosine",
    ),
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=3,  # gbs=1024: 每 epoch 9 步;存点 3/6/9/12/15/18,含 epoch 中间;中点 step9 永久保留
    # save_total_limit 弃用:它驱动 ckpt_manager 的无豁免 prune,会在 step_15 删掉 step_9(审查实锤);滚动清理由 trainer 的带豁免版独管
    save_total_limit=None,
)

data = dict(
    target_cache_path=None,
    train_data_path=[
        "/shenlb/zwf-spec/dataset/prompt-v24/主agent输入_医疗助手_正式环境_promptv24_perfectblend_openai_conv.jsonl",
    ],
    chat_template="v4-flash",
    max_length=45000,
    num_workers=4,
    min_loss_tokens=0,
)

online = dict(
    # 单实例 TP4。注意 v4 原配置注释写"2xtp2 双实例",但 run_svc_opd.sh 实际起的
    # 是 TP=4 单实例 —— 注释与启动脚本不符,别照抄。
    sglang_server_address=["127.0.0.1:30000"],
    request_timeout=1200.0,

    # teacher-forced 预取(opd_rollout=True 时不生效,后者优先)
    feature_prefetch=8,
    feature_prefetch_max=16,

    # 阶段 2:OPD on-policy distillation —— target 自产轨迹 + 同趟特征
    opd_rollout=True,
    opd_max_new_tokens=512,
    opd_temperature=0.0,
    opd_top_p=1.0,
    # v5 默认 4/8(总并发 = 4 rank x 8 = 32):服务 C=8 即饱和吞吐,
    # 更深窗口只加尾延迟与死锁触发面(2026-08-13 实测)
    opd_prefetch=4,
    opd_max_prefetch=8,

    # rollout 结果缓存目录(None = 关)。只存 target 生成的 token ids
    # (每样本约 2 KB),命中后用一次 prefill 重建特征,省掉串行 decode。
    # target 冻结 => 缓存的生成始终是合法 on-policy 样本。
    # 注意:采样参数或权重版本变化会自动失效(已进 key)。
    # v5 默认开(0731 下实测 gbs=64 单步 27.3->5.3 min = 5.2x;gen_ids 每样本约 2KB;
    # key 含采样参数/max_length/权重路径,换任一自动失效)
    opd_rollout_cache_dir="/root/rc_v24",
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join("/root/checkpoints", project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg
