"""train.megatron 配置块的缺省值与解析。

配置文件里 train.megatron 是可选段;缺省时全部取 MEGATRON_TRAIN_DEFAULTS。
注意 --opts 不能创建新 key,想在命令行覆写 megatron 字段,配置文件里必须
先声明 train.megatron 对应字段。
"""

from deepspec.utils.config import to_config_node

MEGATRON_TRAIN_DEFAULTS = dict(
    # 并行度。model.backend="mcore" 时 draft 用 mcore 层实现,可开 TP/EP;
    # 此时这里的值必须与 model.mcore 里的一致(dist.py 有断言,不一致直接报错 ——
    # 因为模型构造中的 init_dspark_parallel 对已建组的情况直接返回,
    # 不一致会让其中一份被静默忽略)。
    # 走原 HF 实现(不设 backend)时仍然只支持 TP=PP=CP=EP=1。
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    # DDP / 梯度归约。
    # use_distributed_optimizer: fp32 master 参数 + Adam 状态按 DP 分片(ZeRO-1),
    #   替代 BF16Optimizer 在每个 rank 复制整份 fp32 状态。
    use_distributed_optimizer=True,
    # grad_reduce_in_fp32: 梯度以 fp32 累积/归约。FSDP 版是 bf16 归约后再转 fp32,
    #   fp32 归约在长梯度累积下数值更稳,代价是梯度 buffer 与通信量翻倍。
    grad_reduce_in_fp32=True,
    # overlap_grad_reduce: mcore 首个迭代会断言所有可训练参数都产生梯度
    #   (bucket golden-count),MoE draft 里长期收不到 token 的 expert 会触发断言,
    #   故缺省关闭;纯稠密 draft(eagle3/qwen3 dspark)可在配置里打开换取通信重叠。
    overlap_grad_reduce=False,
    # overlap_param_gather 与 HF 格式 checkpoint 导出的参数同步性尚未打通,先禁用。
    overlap_param_gather=False,
    check_for_nan_in_grad=False,
    # 集合通信内做 DP 均值,与 FSDP mean all-reduce 的梯度语义对齐。
    average_in_collective=True,
    bucket_size=None,
    # 精度感知优化器(需 distributed optimizer + adam + TE>=2.1)。
    # 19.85B draft 的 fp32 master + Adam m/v = 12 B/param,ZeRO-1 分 4 份后
    # 每卡仍占 55.5 GiB,实测把优化器 step 挤 OOM(140GiB 卡上只剩 25 MiB)。
    # 开启后 store_param_remainders(master 存 16 位余数而非整份 fp32)
    # + bf16 动量,降到 6 B/param ≈ 27.7 GiB/卡。
    use_precision_aware_optimizer=False,
    exp_avg_dtype="fp32",     # "fp32" | "bf16" | "fp16"
    exp_avg_sq_dtype="fp32",  # 二阶动量对动态范围敏感,降精度前先确认 loss 曲线
    # 优化器状态 CPU offload:本项目 128 个 micro-batch 才 step 一次,
    # offload 的 H2D/D2H 开销被摊薄 128 倍,是显存最紧时的兜底手段。
    optimizer_cpu_offload=False,
    optimizer_offload_fraction=1.0,
    overlap_cpu_optimizer_d2h_h2d=False,
    use_torch_optimizer_for_cpu_offload=False,
    # 优化器(与 BF16Optimizer 的 torch.optim.AdamW 缺省一致)。
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    min_lr=0.0,
    lr_decay_style="cosine",
)


def resolve_megatron_config(args):
    """合并 train.megatron 用户配置与缺省值,返回 ConfigNode。"""
    cfg = dict(MEGATRON_TRAIN_DEFAULTS)
    user_cfg = args.train.get("megatron")
    if user_cfg:
        unknown = set(user_cfg) - set(cfg)
        assert not unknown, (
            f"unknown train.megatron keys: {sorted(unknown)}; "
            f"supported: {sorted(cfg)}"
        )
        cfg.update(user_cfg)
    assert not bool(cfg["overlap_param_gather"]), (
        "overlap_param_gather 与 HF 格式 checkpoint 导出的参数同步性未处理,暂不支持"
    )
    return to_config_node(cfg)
