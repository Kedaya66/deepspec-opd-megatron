"""Megatron 训练态 checkpoint。

与 FSDP 版 ckpt_manager 的对齐与差异:
- 模型仍导出 HF 格式(save_pretrained),目录布局、step_latest 软链、
  train_config.py 快照、training_state.rank{r}.pt 命名完全一致,eval.py 无需改动;
- mcore DDP 不切分参数,DP 各 rank 持有完整权重,rank0 直接导出,无需 gather;
- use_distributed_optimizer=True 时 fp32 master/Adam 状态按 DP 分片,
  由 save_parameter_state 聚合到 dp rank0 写单文件 distrib_optim.pt
  (集合调用,所有 rank 必须参与);
- training_state 里带 trainer_backend 标记,防止与 FSDP checkpoint 交叉恢复;
- 磁盘管理(存新前剥旧优化器态、save_total_limit 清理)复用 FSDP 版 helper,
  额外把 megatron 独有的 distrib_optim.pt 按同规则一起剥掉。
"""

import os
import random
import re

import numpy as np
import torch
import torch.distributed as dist

from deepspec.trainer.ckpt_manager import (
    TrainingResumeState,
    _prune_old_checkpoints,
    _rank_training_state_path,
    _strip_prev_optimizer_states,
    save_train_config,
)
from deepspec.utils import (
    ensure_dir,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
    safe_symlink,
)

TRAINER_BACKEND = "megatron"
DISTRIB_OPTIM_STATE_FILE = "distrib_optim.pt"


def _strip_distrib_optim_states(checkpoint_dir_root, keep_dir=None):
    """FSDP 版 helper 只认 training_state.rank*.pt;megatron 的分片优化器态另存
    distrib_optim.pt,这里按同一规则删除(keep_dir 传"最新 step"时保留它)。"""
    if not os.path.isdir(checkpoint_dir_root):
        return
    keep = os.path.abspath(keep_dir) if keep_dir else None
    for name in os.listdir(checkpoint_dir_root):
        p = os.path.join(checkpoint_dir_root, name)
        if not (
            re.fullmatch(r"step_\d+", name)
            and os.path.isdir(p)
            and not os.path.islink(p)
        ):
            continue
        if keep is not None and os.path.abspath(p) == keep:
            continue
        import glob as _g

        for f in _g.glob(os.path.join(p, DISTRIB_OPTIM_STATE_FILE + "*")):
          if os.path.exists(f):
            try:
                os.remove(f)
                print(f"[ckpt] stripped distrib optimizer: {f}", flush=True)
            except OSError:
                pass


def save_megatron_checkpoint(
    *,
    draft_model,
    optimizer,
    scheduler,
    checkpoint_dir_root: str,
    train_config,
    next_micro_step: int,
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
    use_distributed_optimizer: bool,
) -> str:
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    global_step = next_micro_step // gradient_accumulation_steps
    checkpoint_dir = os.path.join(checkpoint_dir_root, f"step_{global_step}")
    if is_global_main_process():
        # 存新前先删旧优化器态,降磁盘峰值(与 FSDP 版 save_checkpoint 一致)
        _strip_prev_optimizer_states(checkpoint_dir_root)
        _strip_distrib_optim_states(checkpoint_dir_root)
        ensure_dir(checkpoint_dir)
        save_train_config(train_config=train_config, checkpoint_dir=checkpoint_dir)
    dist.barrier()

    # ---- 权重导出(2026-09-07 大修)----
    # EP=4 下专家参数按 rank 分片且用本地编号命名(weight0..63 四个 rank 同名不同值),
    # rank0 单点导出会静默丢 192/256 个专家(审查实锤)。导出前按 EP 组 gather 并
    # 重命名为全局编号(ge = ep_rank*n_local + le),与 convert_weights.py 加载切片互逆。
    # gather_object 是集合调用 —— 必须全 rank 参与,放在 main-process 判断之外。
    import re as _re

    from megatron.core import parallel_state as _ps

    _ep_rank = _ps.get_expert_model_parallel_rank()
    _ep_size = _ps.get_expert_model_parallel_world_size()
    _pat = _re.compile(r"^(.*experts\.linear_fc[12]\.weight)(\d+)$")
    _local, _dense = {}, {}
    for key, value in draft_model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            continue  # fp8/TE 的 _extra_state 可能为 None,过滤(09-07 实测崩过)
        m = _pat.match(key)
        v = value.detach().cpu().clone()
        if m and _ep_size > 1:
            n_local_guess = getattr(draft_model.config, "n_routed_experts", None)
            # 全局编号:本地 le -> ge。n_local 由本层实际本地专家数决定,
            # 直接用 le + ep_rank * (本层本地专家数) —— 按同名前缀分组计数。
            _local.setdefault(m.group(1), {})[int(m.group(2))] = v
        else:
            _dense[key] = v
    # 专家分片经磁盘中转汇聚:NCCL 后端的 gather_object 会把 ~4G/rank 的
    # pickle 字节放上显存传输,保存点水位 ~139G 时可能当场 OOM(09-07 核对发现)。
    # 各 rank 写临时分片 -> barrier -> rank0 读合并,零显存零集合通信风险。
    _shard = os.path.join(checkpoint_dir, f".expert_shard.rank{global_rank}.pt")
    torch.save({"ep_rank": _ep_rank, "experts": _local}, _shard)
    dist.barrier()
    if is_global_main_process():
        state_dict = dict(_dense)
        for r in range(world_size):
            part = torch.load(
                os.path.join(checkpoint_dir, f".expert_shard.rank{r}.pt"),
                map_location="cpu", weights_only=False,
            )
            er = part["ep_rank"]
            for prefix, d in part["experts"].items():
                n_local = len(d)
                for le, t in d.items():
                    state_dict[f"{prefix}{er * n_local + le}"] = t
        draft_model.save_pretrained(checkpoint_dir, state_dict=state_dict)
        for r in range(world_size):
            try:
                os.remove(os.path.join(checkpoint_dir, f".expert_shard.rank{r}.pt"))
            except OSError:
                pass

    training_state = {
        "trainer_backend": TRAINER_BACKEND,
        "next_micro_step": int(next_micro_step),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "use_distributed_optimizer": bool(use_distributed_optimizer),
        "global_rank": int(global_rank),
        "world_size": int(world_size),
        "local_batch_size": int(local_batch_size),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    torch.save(
        training_state,
        _rank_training_state_path(checkpoint_dir, global_rank),
    )
    if use_distributed_optimizer:
        # 集合调用。EP=4 时 ChainedOptimizer 的 moe 子优化器组 size=1,
        # 四个 rank 都会写文件 —— 同名并发写必竞态损坏(审查实锤),
        # 改 per-rank 文件;load_parameter_state 侧同规则。
        optimizer.save_parameter_state(
            os.path.join(checkpoint_dir, f"{DISTRIB_OPTIM_STATE_FILE}.rank{global_rank}")
        )
    dist.barrier()
    if is_global_main_process():
        safe_symlink(
            checkpoint_dir,
            os.path.join(checkpoint_dir_root, "step_latest"),
        )
        print_on_global_main(f"Saved checkpoint to {checkpoint_dir}")
        _limit = getattr(getattr(train_config, "logging", None), "save_total_limit", None)
        if _limit:
            # 存完(最新带完整优化器态)+ 软链更新后再清理 → step_latest 永远可安全 resume
            _prune_old_checkpoints(checkpoint_dir_root, int(_limit), strip_old_optimizer=True)
            _strip_distrib_optim_states(checkpoint_dir_root, keep_dir=checkpoint_dir)
    dist.barrier()
    return checkpoint_dir


def load_megatron_training_state(
    *,
    resume_checkpoint_dir: str,
    optimizer,
    scheduler,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
    gradient_accumulation_steps: int,
    micro_batches_per_epoch: int,
    use_distributed_optimizer: bool,
) -> TrainingResumeState:
    state_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    assert os.path.exists(state_path), (
        f"missing per-rank training state: {state_path} "
        "(megatron trainer 断点续训要求 world_size 与保存时一致)"
    )

    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    saved_backend = checkpoint.get("trainer_backend")
    assert saved_backend == TRAINER_BACKEND, (
        f"checkpoint 由 {saved_backend or 'fsdp'} trainer 保存,无法用 megatron "
        "trainer 恢复;换 exp_name 或移除 step_latest 后重跑"
    )
    saved_use_dist_opt = bool(checkpoint["use_distributed_optimizer"])
    assert saved_use_dist_opt == bool(use_distributed_optimizer), (
        "use_distributed_optimizer 与 checkpoint 不一致: "
        f"saved={saved_use_dist_opt}, current={bool(use_distributed_optimizer)}"
    )

    # 与 FSDP 版 load_training_state 一致:load 前校验优化器态有限性,坏就抛错
    # (上层 resume-guard 捕获后回退到"权重热启动 + 全新优化器")
    _opt = checkpoint.get("optimizer", {})
    # EP>1 时 optimizer 是 ChainedOptimizer,state_dict() 返回 list —— .get 会
    # AttributeError(审查实锤)。统一成 list 逐个检查。
    _opts = _opt if isinstance(_opt, list) else [_opt]
    for _o in _opts:
      for _st in ((_o.get("state", {}) if isinstance(_o, dict) else {}) or {}).values():
        if not isinstance(_st, dict):
            continue
        for _v in _st.values():
            if torch.is_tensor(_v) and not torch.isfinite(_v).all():
                raise ValueError("resume optimizer state contains NaN/Inf; refuse to resume")
    optimizer.load_state_dict(_opt)
    scheduler.load_state_dict(checkpoint["scheduler"])
    if use_distributed_optimizer:
        # per-rank 文件(save 侧 09-07 改,防 EP 下并发写竞态);兼容旧单文件
        param_state_path = os.path.join(
            resume_checkpoint_dir, f"{DISTRIB_OPTIM_STATE_FILE}.rank{global_rank}"
        )
        if not os.path.exists(param_state_path):
            param_state_path = os.path.join(resume_checkpoint_dir, DISTRIB_OPTIM_STATE_FILE)
        assert os.path.exists(param_state_path), (
            f"missing distributed optimizer parameter state: {param_state_path}"
        )
        optimizer.load_parameter_state(param_state_path)

    next_micro_step = int(checkpoint["next_micro_step"])
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps."
    )
    assert int(checkpoint["global_rank"]) == int(global_rank)
    assert int(checkpoint["world_size"]) == int(world_size)
    assert int(checkpoint["local_batch_size"]) == int(local_batch_size)

    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state(checkpoint["torch_cuda_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])

    global_step = next_micro_step // gradient_accumulation_steps
    current_epoch = next_micro_step // micro_batches_per_epoch + 1
    print_on_global_main(
        (
            "AUTO-RESUME from "
            f"{resume_checkpoint_dir}, next_micro_step={next_micro_step}, "
            "to force fresh run change exp_name or remove step_latest"
        )
    )
    print_on_local_main(
        f"Resumed from {resume_checkpoint_dir}: "
        f"next_micro_step={next_micro_step}, global_step={global_step}, "
        f"epoch={current_epoch}"
    )
    return TrainingResumeState(next_micro_step=next_micro_step)
