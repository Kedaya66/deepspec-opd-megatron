"""基于 Megatron-Core 的训练器基类。

数据并行用 mcore DistributedDataParallel(梯度 bucket 化 fp32 归约)+
DistributedOptimizer(fp32 master 参数与 Adam 状态按 DP 分片,ZeRO-1),
替代 FSDP 版 BaseTrainer 的 FSDP + BF16Optimizer(每 rank 复制整份 fp32 状态)。

与 FSDP 版对齐的部分(算法子类可直接复用):
- 配置系统、build_models / _build_draft_model / run_batch 接口;
- 数据集构建、StatelessResumableDistributedSampler、CUDAPrefetcher;
- next_micro_step 进度语义、断点续训、suspend、auto-eval、HF 格式 checkpoint。

训练循环换成 megatron 惯用形态:
zero_grad_buffer -> 微批 no_sync 累积 -> finish_grad_sync -> optimizer.step()
(clip 在 step 内部完成,返回 grad_norm)。
"""

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import (
    DistributedDataParallel as MegatronDDP,
    DistributedDataParallelConfig,
)
from megatron.core.transformer import TransformerConfig
from torch.utils.data import DataLoader

from deepspec.data.cuda_prefetcher import CUDAPrefetcher
from deepspec.trainer.base_trainer import (
    BaseTrainer,
    _PRECISION_DTYPES,
    _compute_training_schedule,
    _launch_eval,
)
from deepspec.trainer.ckpt_manager import (
    discover_latest_checkpoint,
    load_resume_draft_model,
)
from deepspec.trainer.megatron.ckpt import (
    load_megatron_training_state,
    save_megatron_checkpoint,
)
from deepspec.trainer.megatron.config import resolve_megatron_config
from deepspec.trainer.megatron.dist import (
    destroy_megatron_parallel_state,
    init_megatron_parallel_state,
)
from deepspec.trainer.megatron.optim import build_optimizer_and_scheduler
from deepspec.utils import (
    StatelessResumableDistributedSampler,
    ensure_dir,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
)
import deepspec.utils.training_logger as training_logger
from deepspec.utils.hfai_suspend import SuspendController


class MegatronBaseTrainer(BaseTrainer):
    # 注意:不要在这里定义 data_collator_cls,组合类
    # (MegatronBaseTrainer 在 MRO 前面)要沿 MRO 取算法子类的定义。

    def __init__(self, local_rank, args):
        self.args = args
        self.megatron_cfg = resolve_megatron_config(args)
        self.device, self.global_rank, self.world_size = init_megatron_parallel_state(
            local_rank,
            self.megatron_cfg,
            model_cfg=args.model,
            seed=int(args.seed) if "seed" in args else 42,
        )
        # TP/PP/CP/EP 均为 1 时 dp == global;批次调度与采样统一走 DP 维度,
        # 后续放开模型并行时这里不需要再改。
        self.dp_rank = parallel_state.get_data_parallel_rank()
        self.dp_world_size = parallel_state.get_data_parallel_world_size()
        self.precision_dtype = _PRECISION_DTYPES[self.args.train.precision]
        self.checkpoint_dir_root = self.args.logging.checkpoint_dir
        self.resume_checkpoint_dir = discover_latest_checkpoint(
            self.checkpoint_dir_root
        )
        self.suspend_controller = SuspendController(device=self.device)
        self.next_micro_step = 0

        if is_global_main_process(): ensure_dir(self.checkpoint_dir_root)
        training_logger.init(
            logging_steps=int(self.args.logging.logging_steps),
            tensorboard_dir=self.args.logging.tensorboard_dir,
        )

        self.draft_model, self.tokenizer = self.build_models()
        if self.resume_checkpoint_dir is not None:
            self.draft_model = load_resume_draft_model(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                draft_model=self.draft_model,
                device=self.device,
                precision_dtype=self.precision_dtype,
                global_rank=self.global_rank,
            )
            # resume 用 from_pretrained 重建了模型,build_models 里开的梯度
            # 检查点不会带过来,不补开会在大 draft 上静默 OOM
            if bool(getattr(self.args.model, "gradient_checkpointing", False)):
                self.draft_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        self.model = self.draft_model
        if self.args.train.torch_compile:
            print_on_local_main("Compiling training model with torch.compile...")
            self.model = torch.compile(self.model, dynamic=True)
        self.model = self._wrap_with_megatron_ddp(self.model)

        self.train_dataset = self._build_train_dataset()
        self._validate_train_dataset()

        (
            self.gradient_accumulation_steps,
            self.samples_per_epoch,
            self.per_rank_samples_per_epoch,
            self.micro_batches_per_epoch,
            self.steps_per_epoch,
            self.max_train_steps,
            self.args.train.num_train_epochs,
        ) = _compute_training_schedule(
            world_size=self.dp_world_size,
            dataset_size=len(self.train_dataset),
            local_batch_size=int(self.args.train.local_batch_size),
            global_batch_size=int(self.args.train.global_batch_size),
            num_train_epochs=int(self.args.train.num_train_epochs),
            max_train_steps=self.args.train.max_train_steps,
        )

        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            ddp_model=self.model,
            precision_dtype=self.precision_dtype,
            lr=self.args.train.lr,
            weight_decay=self.args.train.weight_decay,
            max_grad_norm=self.args.train.max_grad_norm,
            total_steps=self.max_train_steps,
            warmup_ratio=self.args.train.warmup_ratio,
            megatron_cfg=self.megatron_cfg,
        )
        if self.resume_checkpoint_dir is not None:
            try:
                resume_state = load_megatron_training_state(
                    resume_checkpoint_dir=self.resume_checkpoint_dir,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    global_rank=self.global_rank,
                    world_size=self.world_size,
                    local_batch_size=int(self.args.train.local_batch_size),
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    micro_batches_per_epoch=self.micro_batches_per_epoch,
                    use_distributed_optimizer=bool(
                        self.megatron_cfg.use_distributed_optimizer
                    ),
                )
                self.next_micro_step = resume_state.next_micro_step
                print_on_local_main(f"Resumed from {self.resume_checkpoint_dir}")
            except (AssertionError, ValueError, RuntimeError, KeyError, FileNotFoundError, EOFError, AttributeError, TypeError) as _e:  # 09-07: list 型优化器态等结构性差异也要能兜底
                # 优化器态缺失/损坏 → 用该 ckpt 权重热启动 + 全新优化器,绝不 nan/崩
                self.next_micro_step = 0
                print_on_local_main(
                    f"[resume-guard] optimizer resume failed ({_e}); warm-start from ckpt weights, fresh optimizer + dataloader"
                )
        else:
            print_on_local_main("Training from scratch.")
        self.info_board()

    def info_board(self):
        super().info_board()
        print_on_local_main("***** Megatron backend *****")
        print_on_local_main(f"  Data parallel world size = {self.dp_world_size}")
        print_on_local_main(
            f"  Distributed optimizer (ZeRO-1) = {bool(self.megatron_cfg.use_distributed_optimizer)}"
        )
        print_on_local_main(
            f"  Grad reduce in fp32 = {bool(self.megatron_cfg.grad_reduce_in_fp32)}"
        )
        print_on_local_main(
            f"  Overlap grad reduce = {bool(self.megatron_cfg.overlap_grad_reduce)}"
        )

    def _ddp_transformer_config(self):
        """mcore DDP 形参要求 TransformerConfig,但当前版本只读其中
        calculate_per_token_loss 一个字段;给最小合法值即可,与 draft 结构无关。"""
        return TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=1,
            params_dtype=self.precision_dtype,
            bf16=self.precision_dtype is torch.bfloat16,
            fp16=self.precision_dtype is torch.float16,
            calculate_per_token_loss=False,
        )

    def _wrap_with_megatron_ddp(self, model):
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=bool(self.megatron_cfg.grad_reduce_in_fp32),
            overlap_grad_reduce=bool(self.megatron_cfg.overlap_grad_reduce),
            overlap_param_gather=bool(self.megatron_cfg.overlap_param_gather),
            use_distributed_optimizer=bool(
                self.megatron_cfg.use_distributed_optimizer
            ),
            check_for_nan_in_grad=bool(self.megatron_cfg.check_for_nan_in_grad),
            average_in_collective=bool(self.megatron_cfg.average_in_collective),
            bucket_size=self.megatron_cfg.bucket_size,
        )
        print_on_local_main(f"[megatron] ddp_config: {ddp_config}")
        model = MegatronDDP(self._ddp_transformer_config(), ddp_config, model)
        # 各 rank 相同 seed 下初始化应当一致,broadcast 一次兜底(含 buffers)
        model.broadcast_params()
        return model

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            total_size=self.samples_per_epoch,
            start_global_offset_samples=start_offset_samples,
            num_samples=num_samples,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.args.train.local_batch_size),
            sampler=sampler,
            collate_fn=self.data_collator_cls(),
            num_workers=int(self.args.data.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def _current_learning_rate(self):
        return float(self.optimizer.param_groups[0]["lr"])

    def _checkpoint_kwargs(self):
        return dict(
            draft_model=self.draft_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_dir_root=self.checkpoint_dir_root,
            train_config=self.args,
            next_micro_step=self.next_micro_step,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            global_rank=self.global_rank,
            world_size=self.world_size,
            local_batch_size=int(self.args.train.local_batch_size),
            use_distributed_optimizer=bool(
                self.megatron_cfg.use_distributed_optimizer
            ),
        )

    def save_and_eval_checkpoint(self):
        checkpoint_dir = save_megatron_checkpoint(**self._checkpoint_kwargs())
        self._prune_old_checkpoints()
        if is_global_main_process():
            _launch_eval(
                target_model_name_or_path=self.args.model.target_model_name_or_path,
                checkpoint_dir=checkpoint_dir,
                step=self.global_step,
                tensorboard_dir=self.args.logging.tensorboard_dir,
                exp_name=self.args.exp_name,
            )
        dist.barrier()
        return checkpoint_dir

    def _prune_old_checkpoints(self, keep_last: int = 2):
        """滚动只留最新 keep_last 个 step_* 目录;全程中点那份永久豁免。

        2026-08-28 用户要求:权重只留最新两个 + 保留最中间那步(epoch 边界)。
        旧优化器态已由 _strip_prev_optimizer_states 处理,这里删旧模型权重目录。
        仅 global rank0 执行。
        """
        import os
        import re
        import shutil

        if dist.is_initialized() and dist.get_rank() != 0:
            return
        root = self._checkpoint_kwargs()["checkpoint_dir_root"]
        mid_step = self.max_train_steps // 2  # 全程中点(2 epoch 时=epoch1 末)
        try:
            steps = sorted(
                int(m.group(1))
                for d in os.listdir(root)
                if (m := re.fullmatch(r"step_(\d+)", d))
            )
        except FileNotFoundError:
            return
        for st in steps[:-keep_last] if keep_last else steps:
            if st == mid_step:
                continue  # 中点豁免
            shutil.rmtree(os.path.join(root, f"step_{st}"), ignore_errors=True)
            print_on_global_main(
                f"[ckpt] 滚动清理 step_{st}(留最新{keep_last}+中点 step_{mid_step})"
            )

    def _save_and_suspend(self):
        print_on_global_main("Saving checkpoint before suspending...")
        save_megatron_checkpoint(**self._checkpoint_kwargs())
        dist.barrier()
        if is_global_main_process():
            print_on_global_main("Going to suspend...")
            self.suspend_controller.go_suspend()
        dist.barrier()

    def train(self):
        self.model.train()
        if self.global_step >= self.max_train_steps:
            return

        local_batch_size = int(self.args.train.local_batch_size)
        total_micro_steps = self.max_train_steps * self.gradient_accumulation_steps
        remaining_micro_steps = total_micro_steps - self.next_micro_step
        remaining_samples = remaining_micro_steps * local_batch_size

        dataloader = self._build_train_dataloader(
            start_offset_samples=self.next_micro_step * local_batch_size,
            num_samples=remaining_samples,
        )
        prefetcher = CUDAPrefetcher(dataloader, self.device)
        training_logger.start_session(global_step=self.global_step)

        self.optimizer.zero_grad()
        self.model.zero_grad_buffer()

        with self.suspend_controller.monitoring():
            for batch in prefetcher:
                should_sync = (
                    (self.next_micro_step + 1) % self.gradient_accumulation_steps == 0
                )
                sync_context = nullcontext() if should_sync else self.model.no_sync()
                with sync_context:
                    loss = self.run_batch(batch) / self.gradient_accumulation_steps
                    if torch.isfinite(loss):
                        loss.backward()
                    else:
                        print_on_local_main(
                            f"[nan-guard] non-finite loss={loss.item()}, skip this micro-batch backward"
                        )
                self.next_micro_step += 1

                if not should_sync:
                    continue

                self.model.finish_grad_sync()
                # 先推进 LR 再 step,对齐 FSDP 版 torch scheduler 的一步语义
                # (第 k 个 optimizer step 用 k/warmup_steps 比例的 lr)
                self.scheduler.step(increment=1)
                update_successful, grad_norm, _ = self.optimizer.step()
                if not update_successful:
                    # bf16 无 loss scaler,正常不会走到;fp16 溢出时跳过该步
                    print_on_global_main(
                        f"[megatron] optimizer skipped step {self.global_step} "
                        "(inf/nan found in grads)"
                    )
                self.optimizer.zero_grad()
                self.model.zero_grad_buffer()

                training_logger.on_optimizer_step(
                    global_step=self.global_step,
                    next_micro_step=self.next_micro_step,
                    micro_batches_per_epoch=self.micro_batches_per_epoch,
                    max_train_steps=self.max_train_steps,
                    learning_rate=self._current_learning_rate(),
                    grad_norm=float(grad_norm) if grad_norm is not None else 0.0,
                )

                if self.global_step % int(self.args.logging.checkpointing_steps) == 0:
                    self.save_and_eval_checkpoint()

                if self.suspend_controller.requested():
                    self._save_and_suspend()
                    return

        latest = os.path.join(self._checkpoint_kwargs()["checkpoint_dir_root"], "step_latest")
        already = os.path.islink(latest) and os.readlink(latest).endswith(f"step_{self.global_step}")
        if already:
            print_on_global_main(f"[ckpt] step_{self.global_step} 已存(循环内),收尾跳过重复保存")
        elif self.next_micro_step % self.gradient_accumulation_steps == 0:
            self.save_and_eval_checkpoint()
        else:
            # 2026-08-13:数据流异常提前结束(如特征服务不可用)时会停在
            # accumulation 中间。此时 checkpoint 断言必炸;跳过并醒目告警,
            # 保留日志现场比二次崩溃有用。
            print_on_global_main(
                f"[megatron] WARNING: 训练在 accumulation 中途结束"
                f"(next_micro_step={self.next_micro_step}),跳过收尾 checkpoint"
            )

    def clean_up(self):
        training_logger.close()
        dist.barrier()
        destroy_megatron_parallel_state()
        dist.destroy_process_group()
