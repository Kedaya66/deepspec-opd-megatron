from contextlib import contextmanager, nullcontext
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor, Shard
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from deepspec.data import CacheDataset, validate_train_cache
from deepspec.data.cuda_prefetcher import CUDAPrefetcher
from deepspec.utils import (
    BF16Optimizer,
    StatelessResumableDistributedSampler,
    ensure_dir,
    init_dist,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
)
from deepspec.trainer.ckpt_manager import (
    discover_latest_checkpoint,
    load_resume_draft_model,
    load_training_state,
    save_checkpoint,
)
import deepspec.utils.training_logger as training_logger
from deepspec.utils.hfai_suspend import SuspendController


_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_SHARDING_STRATEGIES = {
    "full_shard": dict(hybrid=False, ddp=False, reshard_after_forward=True),
    "shard_grad_op": dict(hybrid=False, ddp=False, reshard_after_forward=False),
    "no_shard": dict(hybrid=False, ddp=True, reshard_after_forward=False),
    "hybrid_shard": dict(hybrid=True, ddp=False, reshard_after_forward=True),
    "hybrid_shard_zero2": dict(hybrid=True, ddp=False, reshard_after_forward=False),
    "_hybrid_shard_zero2": dict(hybrid=True, ddp=False, reshard_after_forward=False),
}


def _build_fsdp_kwargs(
    *, sharding_strategy_name: str, precision_dtype, world_size: int,
    cpu_offload: bool = False,
) -> dict:
    strategy = _SHARDING_STRATEGIES[sharding_strategy_name]
    fsdp_kwargs = dict(
        reshard_after_forward=strategy["reshard_after_forward"],
        mp_policy=MixedPrecisionPolicy(param_dtype=precision_dtype),
    )
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    if strategy["ddp"]:
        fsdp_kwargs["mesh"] = init_device_mesh(
            "cuda",
            (world_size, 1),
            mesh_dim_names=("replicate", "shard"),
        )
    elif strategy["hybrid"]:
        devices_per_node = torch.cuda.device_count()
        assert world_size % devices_per_node == 0
        fsdp_kwargs["mesh"] = init_device_mesh(
            "cuda",
            (world_size // devices_per_node, devices_per_node),
            mesh_dim_names=("replicate", "shard"),
        )
    return fsdp_kwargs


@contextmanager
def _fsdp2_no_sync(model):
    """Match FSDP1 no_sync semantics during gradient accumulation."""
    model.set_requires_gradient_sync(False)
    try:
        yield
    finally:
        model.set_requires_gradient_sync(True)


def _clip_grad_norm_fsdp2(parameters, max_norm: float) -> torch.Tensor:
    """Clip local FSDP2 shards with one norm collective, including CPU offload.

    Applying torch.nn.utils.clip_grad_norm_ directly to a list of DTensors can
    dispatch per-tensor distributed operations. This computes norms on local
    shards and performs one reduction over the mesh's shard dimension.
    """
    local_grads = []
    shard_group = None
    norm_device = None
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        if isinstance(grad, DTensor):
            local_grad = grad.to_local()
            for mesh_dim, placement in enumerate(grad.placements):
                if isinstance(placement, Shard):
                    shard_group = grad.device_mesh.get_group(mesh_dim)
                    break
        else:
            local_grad = grad
        local_grads.append(local_grad.detach())
        norm_device = local_grad.device

    if not local_grads:
        return torch.zeros((), device=norm_device or "cpu")

    norms = torch._foreach_norm(local_grads, 2.0)
    total_sq = torch.stack([norm.float().square() for norm in norms]).sum()
    if shard_group is not None:
        reduce_value = total_sq
        if dist.get_backend(shard_group) == "nccl" and not reduce_value.is_cuda:
            reduce_value = reduce_value.to(torch.cuda.current_device())
        dist.all_reduce(reduce_value, op=dist.ReduceOp.SUM, group=shard_group)
        total_sq = reduce_value.to(norm_device)
    total_norm = total_sq.sqrt()
    clip_coef = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(local_grads, clip_coef.to(norm_device))
    return total_norm


def _compute_gradient_accumulation_steps(
    *, world_size: int, local_batch_size: int, global_batch_size: int
) -> int:
    denom = world_size * local_batch_size
    assert global_batch_size % denom == 0, (
        "global_batch_size must be divisible by world_size * local_batch_size: "
        f"global_batch_size={global_batch_size}, world_size={world_size}, "
        f"local_batch_size={local_batch_size}"
    )
    return global_batch_size // denom


def _compute_samples_per_epoch(*, dataset_size: int, global_batch_size: int) -> int:
    samples_per_epoch = (dataset_size // global_batch_size) * global_batch_size
    assert samples_per_epoch > 0, (
        "train dataset is too small to form one full global batch: "
        f"dataset_size={dataset_size}, global_batch_size={global_batch_size}"
    )
    return samples_per_epoch


def _compute_training_schedule(
    *,
    world_size: int,
    dataset_size: int,
    local_batch_size: int,
    global_batch_size: int,
    num_train_epochs: int,
    max_train_steps=None,
) -> tuple[int, int, int, int, int, int, int]:
    gradient_accumulation_steps = _compute_gradient_accumulation_steps(
        world_size=world_size,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
    )
    samples_per_epoch = _compute_samples_per_epoch(
        dataset_size=dataset_size,
        global_batch_size=global_batch_size,
    )
    per_rank_samples_per_epoch = samples_per_epoch // world_size
    micro_batches_per_epoch = per_rank_samples_per_epoch // local_batch_size
    steps_per_epoch = micro_batches_per_epoch // gradient_accumulation_steps
    if max_train_steps is None:
        resolved_max_train_steps = int(num_train_epochs) * steps_per_epoch
        resolved_num_train_epochs = int(num_train_epochs)
    else:
        resolved_max_train_steps = int(max_train_steps)
        resolved_num_train_epochs = math.ceil(
            resolved_max_train_steps / steps_per_epoch
        )
    return (
        gradient_accumulation_steps,
        samples_per_epoch,
        per_rank_samples_per_epoch,
        micro_batches_per_epoch,
        steps_per_epoch,
        resolved_max_train_steps,
        resolved_num_train_epochs,
    )


def _launch_eval(
    *,
    target_model_name_or_path: str,
    checkpoint_dir: str,
    step: int,
    tensorboard_dir: str,
    exp_name: str,
) -> None:
    from deepspec.utils.constant import auto_eval_command
    if auto_eval_command is not None:
        command = auto_eval_command(target_model_name_or_path,checkpoint_dir,step,tensorboard_dir,exp_name)
        print_on_global_main(f"Submitting auto eval for {checkpoint_dir}")
        print_on_global_main(command)
        os.system(command)
    else:
        print("You can use this function to launch your auto eval script!")

class BaseTrainer:
    data_collator_cls = None

    def __init__(self, local_rank, args):
        self.args = args
        self.device, self.global_rank, self.world_size = init_dist(local_rank)
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
            layer_classes = self._fsdp_transformer_layer_classes()
            if layer_classes:
                compiled = 0
                for module in self.model.modules():
                    if type(module) in layer_classes:
                        module.compile(dynamic=True)
                        compiled += 1
                print_on_local_main(
                    f"Compiling training model with torch.compile... "
                    f"(per-layer, {compiled} layers)"
                )
            else:
                print_on_local_main("Compiling training model with torch.compile...")
                self.model.compile(dynamic=True)
        self.model = self._wrap_with_fsdp(self.model)

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
            world_size=self.world_size,
            dataset_size=len(self.train_dataset),
            local_batch_size=int(self.args.train.local_batch_size),
            global_batch_size=int(self.args.train.global_batch_size),
            num_train_epochs=int(self.args.train.num_train_epochs),
            max_train_steps=self.args.train.max_train_steps,
        )

        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=float(self.args.train.lr),
            total_steps=self.max_train_steps,
            warmup_ratio=float(self.args.train.warmup_ratio),
            weight_decay=float(self.args.train.weight_decay),
        )
        if self.resume_checkpoint_dir is not None:
            try:
                resume_state = load_training_state(
                    resume_checkpoint_dir=self.resume_checkpoint_dir,
                    optimizer=self.optimizer,
                    global_rank=self.global_rank,
                    world_size=self.world_size,
                    local_batch_size=int(self.args.train.local_batch_size),
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    micro_batches_per_epoch=self.micro_batches_per_epoch,
                )
                self.next_micro_step = resume_state.next_micro_step
                print_on_local_main(f"Resumed from {self.resume_checkpoint_dir}")
            except (AssertionError, ValueError, RuntimeError, KeyError, FileNotFoundError, EOFError) as _e:
                # 优化器态缺失/损坏 → 用该 ckpt 权重热启动 + 全新优化器,绝不 nan/崩
                self.next_micro_step = 0
                print_on_local_main(
                    f"[resume-guard] optimizer resume failed ({_e}); warm-start from ckpt weights, fresh optimizer + dataloader"
                )
        else:
            print_on_local_main("Training from scratch.")
        self.info_board()

    @property
    def global_step(self):
        return self.next_micro_step // self.gradient_accumulation_steps

    def info_board(self):
        print_on_local_main("***** Running training *****")
        print_on_local_main(f"  Train dataset size = {len(self.train_dataset)}")
        print_on_local_main(f"  Num train epochs = {self.args.train.num_train_epochs}")
        print_on_local_main(f"  Samples per epoch = {self.samples_per_epoch}")
        print_on_local_main(f"  Local batch size = {self.args.train.local_batch_size}")
        print_on_local_main(f"  Global batch size = {self.args.train.global_batch_size}")
        print_on_local_main(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        print_on_local_main(f"  Steps per epoch = {self.steps_per_epoch}")
        print_on_local_main(f"  Max train steps = {self.max_train_steps}")

    def build_models(self):
        model_args = self.args.model

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
        )

        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)

        # Training only uses the target checkpoint to initialize frozen draft
        # embeddings and lm_head weights.
        target_model = AutoModelForCausalLM.from_pretrained(
            model_args.target_model_name_or_path,
            dtype=self.precision_dtype,
        ).to(device="cpu").eval()
        target_embed_tokens = target_model.get_input_embeddings()
        target_lm_head = target_model.get_output_embeddings()
        assert (target_lm_head is not None) and (target_embed_tokens is not None)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )
        del target_model
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        raise NotImplementedError

    def _build_train_dataset(self):
        """默认:离线 target cache 数据集。在线训练子类覆写为 token 数据集。"""
        return CacheDataset(cache_dir=self.args.data.target_cache_path)

    def _validate_train_dataset(self):
        """默认:校验离线 cache 与 draft 配置匹配。在线训练子类覆写为 no-op。"""
        validate_train_cache(
            train_dataset=self.train_dataset,
            draft_model=self.draft_model,
            target_model_name_or_path=self.args.model.target_model_name_or_path,
        )

    def _fsdp_transformer_layer_classes(self):
        """从 draft 的 _no_split_modules 解析出实际 layer 类,用于逐层 FSDP 包裹。"""
        names = set(getattr(self.draft_model, "_no_split_modules", []) or [])
        if not names:
            return None
        classes = set()
        for module in self.draft_model.modules():
            if type(module).__name__ in names:
                classes.add(type(module))
        return classes or None

    def _wrap_with_fsdp(self, model):
        fsdp_kwargs = _build_fsdp_kwargs(
            sharding_strategy_name=self.args.train.sharding_strategy,
            precision_dtype=self.precision_dtype,
            world_size=self.world_size,
            cpu_offload=bool(getattr(self.args.train, "cpu_offload", False)),
        )
        # FSDP2 has no auto-wrap policy: shard each transformer layer first,
        # then shard the root to cover embeddings, heads, and norms.
        layer_classes = self._fsdp_transformer_layer_classes()
        if layer_classes:
            layers = [module for module in model.modules() if type(module) in layer_classes]
            for layer in layers:
                fully_shard(layer, **fsdp_kwargs)
            print_on_local_main(
                f"[fsdp2] per-layer fully_shard on "
                f"{sorted(c.__name__ for c in layer_classes)} ({len(layers)} layers)"
            )
        root_kwargs = dict(fsdp_kwargs)
        # The root owns the terminal confidence/head path. Keeping it
        # unsharded between forward and backward avoids FSDP2 invalidating the
        # activation storage before that path's backward hook consumes it.
        root_kwargs["reshard_after_forward"] = False
        fully_shard(model, **root_kwargs)
        assert isinstance(model, FSDPModule)
        return model

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
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

    def run_batch(self, batch):
        raise NotImplementedError

    def _checkpoint_kwargs(self):
        return dict(
            model=self.model,
            draft_model=self.draft_model,
            optimizer=self.optimizer,
            checkpoint_dir_root=self.checkpoint_dir_root,
            train_config=self.args,
            next_micro_step=self.next_micro_step,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            global_rank=self.global_rank,
            world_size=self.world_size,
            local_batch_size=int(self.args.train.local_batch_size),
        )

    def save_and_eval_checkpoint(self):
        checkpoint_dir = save_checkpoint(**self._checkpoint_kwargs())
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

    def _save_and_suspend(self):
        print_on_global_main("Saving checkpoint before suspending...")
        save_checkpoint(**self._checkpoint_kwargs())
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

        with self.suspend_controller.monitoring():
            for batch in prefetcher:
                should_sync = (
                    (self.next_micro_step + 1) % self.gradient_accumulation_steps == 0
                )
                sync_context = nullcontext() if should_sync else _fsdp2_no_sync(self.model)
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

                clip_start = time.perf_counter()
                grad_norm = _clip_grad_norm_fsdp2(
                    self.model.parameters(),
                    float(self.args.train.max_grad_norm),
                )
                optimizer_start = time.perf_counter()
                self.optimizer.step()
                optimizer_end = time.perf_counter()
                print_on_local_main(
                    f"[timing] clip={optimizer_start - clip_start:.1f}s "
                    f"optim={optimizer_end - optimizer_start:.1f}s"
                )
                training_logger.on_optimizer_step(
                    global_step=self.global_step,
                    next_micro_step=self.next_micro_step,
                    micro_batches_per_epoch=self.micro_batches_per_epoch,
                    max_train_steps=self.max_train_steps,
                    learning_rate=self.optimizer.get_learning_rate(),
                    grad_norm=grad_norm.item(),
                )

                if self.global_step % int(self.args.logging.checkpointing_steps) == 0:
                    self.save_and_eval_checkpoint()

                if self.suspend_controller.requested():
                    self._save_and_suspend()
                    return

        self.save_and_eval_checkpoint()

    def clean_up(self):
        training_logger.close()
        dist.barrier()
        dist.destroy_process_group()
