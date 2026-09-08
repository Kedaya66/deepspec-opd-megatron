"""Megatron 优化器与 LR 调度器构建。

对齐 FSDP 版 BF16Optimizer(AdamW + CosineAnnealingWarmupLR)的训练语义:
- weight decay 应用到全部可训练参数(get_megatron_optimizer 传
  config_overrides=None 时不做 bias/1D 参数豁免,与 AdamW 缺省一致);
- 线性 warmup 到 lr,余下步数 cosine 退火到 min_lr(缺省 0);
  warmup 段与 torch 版逐步精确一致(需配合 trainer 里"先 scheduler.step(1)
  再 optimizer.step"的调用顺序);cosine 段与 torch 版差一步相位
  (torch 链式调度器交接后 cosine 索引按 k-1-warmup 计,本实现按 k-warmup),
  200 步日程下最大偏差约 0.8% 峰值 lr,数千步真实训练为千分位级,可忽略;
- 梯度裁剪由 optimizer.step() 内部完成(clip_grad=train.max_grad_norm),
  step 返回 (update_successful, grad_norm, num_zeros_in_grad)。
"""

import torch
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _dtype(name):
    key = str(name).lower()
    assert key in _DTYPES, f"unsupported dtype {name!r}; use one of {sorted(_DTYPES)}"
    return _DTYPES[key]


def build_optimizer_and_scheduler(
    *,
    ddp_model,
    precision_dtype,
    lr,
    weight_decay,
    max_grad_norm,
    total_steps,
    warmup_ratio,
    megatron_cfg,
):
    lr = float(lr)
    min_lr = float(megatron_cfg.min_lr)
    weight_decay = float(weight_decay)
    total_steps = int(total_steps)

    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        adam_beta1=float(megatron_cfg.adam_beta1),
        adam_beta2=float(megatron_cfg.adam_beta2),
        adam_eps=float(megatron_cfg.adam_eps),
        bf16=precision_dtype is torch.bfloat16,
        fp16=precision_dtype is torch.float16,
        params_dtype=precision_dtype,
        use_distributed_optimizer=bool(megatron_cfg.use_distributed_optimizer),
        overlap_param_gather=bool(megatron_cfg.overlap_param_gather),
        clip_grad=float(max_grad_norm),
        log_num_zeros_in_grad=False,
        # 降低优化器状态显存:master 参数存 16 位余数 + 动量可降到 bf16。
        # 见 config.py 中该组字段的注释(不开时全部保持 fp32,与原行为一致)。
        use_precision_aware_optimizer=bool(
            megatron_cfg.use_precision_aware_optimizer
        ),
        exp_avg_dtype=_dtype(megatron_cfg.exp_avg_dtype),
        exp_avg_sq_dtype=_dtype(megatron_cfg.exp_avg_sq_dtype),
        optimizer_cpu_offload=bool(megatron_cfg.optimizer_cpu_offload),
        optimizer_offload_fraction=float(megatron_cfg.optimizer_offload_fraction),
        overlap_cpu_optimizer_d2h_h2d=bool(
            megatron_cfg.overlap_cpu_optimizer_d2h_h2d
        ),
        use_torch_optimizer_for_cpu_offload=bool(
            megatron_cfg.use_torch_optimizer_for_cpu_offload
        ),
    )
    optimizer = get_megatron_optimizer(opt_config, [ddp_model])

    warmup_steps = int(float(warmup_ratio) * total_steps)
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=lr,
        min_lr=min_lr,
        lr_warmup_steps=warmup_steps,
        lr_decay_steps=total_steps,
        lr_decay_style=str(megatron_cfg.lr_decay_style),
        start_wd=weight_decay,
        end_wd=weight_decay,
        wd_incr_steps=total_steps,
        wd_incr_style="constant",
    )
    return optimizer, scheduler
