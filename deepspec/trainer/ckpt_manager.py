import os
import re
import random
import shutil
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import yaml
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from deepspec.utils import (
    ensure_dir,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
    safe_symlink,
)


TRAIN_CONFIG_FILE_NAME = "train_config.py"


def _prune_old_checkpoints(checkpoint_dir_root, keep, strip_old_optimizer=False):
    """保留最新 keep 个 step 目录(更旧的整目录删除)。
    strip_old_optimizer=True 时,再把保留下来的旧目录里的 training_state.rank*.pt 删掉(只留权重),
    这样只有稍后写入的"最新 step"带完整优化器态、可 resume;旧的仅供 eval。"""
    if keep is None or keep < 0:
        return
    steps = []
    for name in os.listdir(checkpoint_dir_root):
        m = re.fullmatch(r"step_(\d+)", name)
        p = os.path.join(checkpoint_dir_root, name)
        if m and os.path.isdir(p) and not os.path.islink(p):
            steps.append((int(m.group(1)), p))
    steps.sort(key=lambda x: x[0], reverse=True)
    for _, path in steps[keep:]:
        try:
            shutil.rmtree(path); print(f"[ckpt] pruned old checkpoint: {path}", flush=True)
        except OSError as e:
            print(f"[ckpt] prune failed {path}: {e}", flush=True)
    if strip_old_optimizer:
        for _, path in steps[1:keep]:  # 保留集里除"最新"外都删优化器态(只留权重)
            for f in os.listdir(path):
                if f.startswith("training_state.rank") and f.endswith(".pt"):
                    try:
                        os.remove(os.path.join(path, f)); print(f"[ckpt] stripped optimizer: {path}/{f}", flush=True)
                    except OSError: pass


def discover_latest_checkpoint(checkpoint_dir):
    latest_link = os.path.join(checkpoint_dir, "step_latest")
    if not (os.path.islink(latest_link) or os.path.isdir(latest_link)):
        return None
    return os.path.realpath(latest_link)


def save_train_config(*, train_config, checkpoint_dir: str) -> str:
    dest_path = os.path.join(checkpoint_dir, TRAIN_CONFIG_FILE_NAME)
    if not is_global_main_process():
        return dest_path

    ensure_dir(checkpoint_dir)
    shutil.copy(train_config._origin_config_path, dest_path)
    opts = train_config._origin_opts
    if opts:
        with open(dest_path, "a", encoding="utf-8") as handle:
            handle.write("\n\n# --opts overrides applied at save time\n")
            for opt in opts:
                handle.write(_render_opt_assignment(opt) + "\n")
    return dest_path


def _render_opt_assignment(opt: str) -> str:
    key, raw_value = opt.split("=", 1)
    head, *rest = key.split(".")
    accessors = "".join(f"[{part!r}]" for part in rest)
    value = yaml.safe_load(raw_value)
    return f"{head}{accessors} = {value!r}"


@dataclass(frozen=True)
class TrainingResumeState:
    # next_micro_step is the single source of truth for training progress;
    # global_step and current_epoch are derived from it together with
    # gradient_accumulation_steps / micro_batches_per_epoch.
    next_micro_step: int


def load_resume_draft_model(
    *,
    resume_checkpoint_dir: str,
    draft_model,
    device,
    precision_dtype,
    global_rank: int,
):
    state_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    assert os.path.exists(state_path)
    model_path = os.path.join(resume_checkpoint_dir, "model.safetensors")
    if (
        type(draft_model).__name__.endswith("DSparkModelMatchV1")
        and os.path.exists(model_path)
    ):
        _load_match_v1_training_weights(draft_model, model_path, device=device)
        draft_model.set_embedding_head_trainable(False)
        return draft_model

    # mcore 路线:ckpt 的 config.json(save_pretrained 产物)不含 mcore 字段
    # (use_transformer_engine/moe_grouped_gemm 等),from_pretrained 重新构造会
    # 拿默认值撞校验(2026-09-07 实测 resume 必炸)。改为与 MatchV1 同模式:
    # 不重构造,直接把 safetensors 权重灌进按训练配置建好的 draft_model。
    if "Mcore" in type(draft_model).__name__ or hasattr(draft_model, "mcore_config"):
        import glob as _glob

        from safetensors.torch import load_file as _load_sf

        import re as _re

        from megatron.core import parallel_state as _ps

        sd_raw = {}
        for f in sorted(_glob.glob(os.path.join(resume_checkpoint_dir, "*.safetensors"))):
            sd_raw.update(_load_sf(f))
        assert sd_raw, f"resume 目录无 safetensors: {resume_checkpoint_dir}"
        # ckpt 里专家键是全局编号(save 侧 gather 后重命名);灌入前按本 rank
        # 的 EP 分片切回本地编号,非本 rank 的专家键跳过。
        _pat = _re.compile(r"^(.*experts\.linear_fc[12]\.weight)(\d+)$")
        _ep_rank = _ps.get_expert_model_parallel_rank()
        _ep_size = _ps.get_expert_model_parallel_world_size()
        _tot = {}
        for k in sd_raw:
            m = _pat.match(k)
            if m:
                _tot[m.group(1)] = max(_tot.get(m.group(1), 0), int(m.group(2)) + 1)
        sd = {}
        for k, v in sd_raw.items():
            m = _pat.match(k)
            if m and _ep_size > 1:
                n_local = _tot[m.group(1)] // _ep_size
                ge = int(m.group(2))
                er, le = divmod(ge, n_local)
                if er != _ep_rank:
                    continue
                sd[f"{m.group(1)}{le}"] = v
            else:
                sd[k] = v
        missing, unexpected = draft_model.load_state_dict(sd, strict=False)
        # extra_state(TE fp8)存时被过滤,missing 中仅允许该类键
        bad = [k for k in missing if "_extra_state" not in k]
        assert not bad, f"resume 缺权重键: {bad[:5]}"
        assert not unexpected, f"resume 多余键: {list(unexpected)[:5]}"
        draft_model.set_embedding_head_trainable(False)
        return draft_model

    resumed_model = type(draft_model).from_pretrained(
        resume_checkpoint_dir,
        dtype=precision_dtype,
        attn_implementation=str(draft_model.config._attn_implementation),
    )
    resumed_model = resumed_model.to(device=device, dtype=precision_dtype)
    resumed_model.set_embedding_head_trainable(False)
    return resumed_model


def _match_v1_training_key(saved_key: str) -> str:
    """Map save_pretrained's DSpark serving names back to training names.

    This is a pure rename. In particular, it must not apply the official
    checkpoint loader's RoPE permutation because trained checkpoints are
    already in the model's internal layout.
    """
    if saved_key == "embed.weight":
        return "embed_tokens.weight"
    if saved_key == "head.weight":
        return "lm_head.weight"

    match = re.match(r"^layers\.(\d+)\.(.+)$", saved_key)
    if match is None:
        return saved_key
    layer, suffix = match.groups()
    if suffix.startswith("attn."):
        suffix = "self_attn." + suffix[len("attn.") :]
    elif suffix.startswith("attn_norm."):
        suffix = "input_layernorm." + suffix[len("attn_norm.") :]
    elif suffix.startswith("ffn_norm."):
        suffix = "post_attention_layernorm." + suffix[len("ffn_norm.") :]
    elif suffix.startswith("ffn."):
        suffix = "mlp." + suffix[len("ffn.") :]
    elif suffix.startswith("hc_attn_"):
        suffix = "attn_hc." + suffix[len("hc_attn_") :]
    elif suffix.startswith("hc_ffn_"):
        suffix = "ffn_hc." + suffix[len("hc_ffn_") :]
    return f"layers.{layer}.{suffix}"


def _load_match_v1_training_weights(model, model_path: str, *, device) -> None:
    model_state = model.state_dict()
    loaded = set()
    unexpected = []
    with torch.no_grad(), safe_open(model_path, framework="pt", device="cpu") as source:
        for saved_key in source.keys():
            model_key = _match_v1_training_key(saved_key)
            target = model_state.get(model_key)
            if target is None:
                unexpected.append((saved_key, model_key))
                continue
            value = source.get_tensor(saved_key)
            assert tuple(value.shape) == tuple(target.shape), (
                f"resume shape mismatch for {model_key}: "
                f"checkpoint={tuple(value.shape)}, model={tuple(target.shape)}"
            )
            target.copy_(value.to(device=device, dtype=target.dtype))
            loaded.add(model_key)

    missing = sorted(set(model_state) - loaded)
    assert not unexpected, f"unexpected resume keys: {unexpected[:10]}"
    assert not missing, f"missing resume keys: {missing[:10]}"
    print_on_local_main(
        f"Loaded match-v1 training weights from {model_path} ({len(loaded)} tensors)"
    )


def load_training_state(
    *,
    resume_checkpoint_dir: str,
    optimizer,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
    gradient_accumulation_steps: int,
    micro_batches_per_epoch: int,
) -> TrainingResumeState:
    state_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    assert os.path.exists(state_path)

    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    _opt = checkpoint.get("optimizer", {})
    optimizer.load_state_dict(_opt)
    _validate_optimizer_state_finite(optimizer)

    next_micro_step = int(checkpoint["next_micro_step"])
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps."
    )

    saved_rank = int(checkpoint["global_rank"])
    assert saved_rank == int(global_rank)
    
    saved_world_size = int(checkpoint["world_size"])
    assert saved_world_size == int(world_size)
    
    saved_local_batch_size = int(checkpoint["local_batch_size"])
    assert saved_local_batch_size == int(local_batch_size)

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


def _validate_optimizer_state_finite(optimizer) -> None:
    """Validate loaded optimizer shards on their runtime device in foreach batches."""
    inner_optimizer = getattr(optimizer, "optimizer", optimizer)
    groups = {}
    for state in inner_optimizer.state.values():
        for value in state.values():
            if not torch.is_tensor(value) or not value.is_floating_point():
                continue
            local = value.to_local() if isinstance(value, DTensor) else value
            groups.setdefault((local.device, local.dtype), []).append(local)

    for tensors in groups.values():
        norms = torch._foreach_norm(tensors, 2.0)
        if not torch.stack([norm.float() for norm in norms]).isfinite().all():
            raise ValueError("resume optimizer state contains NaN/Inf; refuse to resume")


def _strip_prev_optimizer_states(checkpoint_dir_root):
    """存新 step 前,删除旧 step 的 rank optimizer state,只保留模型权重。"""
    if not os.path.isdir(checkpoint_dir_root):
        return
    for name in os.listdir(checkpoint_dir_root):
        path = os.path.join(checkpoint_dir_root, name)
        if not re.fullmatch(r"step_\d+", name) or not os.path.isdir(path) or os.path.islink(path):
            continue
        for filename in os.listdir(path):
            if filename.startswith("training_state.rank") and filename.endswith(".pt"):
                try:
                    os.remove(os.path.join(path, filename))
                    print(f"[ckpt] pre-strip optimizer: {path}/{filename}", flush=True)
                except OSError:
                    pass



def save_checkpoint(
    *,
    model,
    draft_model,
    optimizer,
    checkpoint_dir_root: str,
    train_config,
    next_micro_step: int,
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
) -> str:
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    global_step = next_micro_step // gradient_accumulation_steps
    checkpoint_dir = os.path.join(checkpoint_dir_root, f"step_{global_step}")
    if is_global_main_process():
        _strip_prev_optimizer_states(checkpoint_dir_root)
        ensure_dir(checkpoint_dir)
        save_train_config(train_config=train_config, checkpoint_dir=checkpoint_dir)
    dist.barrier()
    _save_model_checkpoint(
        model=model,
        draft_model=draft_model,
        checkpoint_dir=checkpoint_dir,
    )
    training_state = _serialize_training_state(
        optimizer=optimizer,
        next_micro_step=next_micro_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_rank=global_rank,
        world_size=world_size,
        local_batch_size=local_batch_size,
    )
    torch.save(
        training_state,
        _rank_training_state_path(checkpoint_dir, global_rank),
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
            _prune_old_checkpoints(checkpoint_dir_root, int(_limit), strip_old_optimizer=True)
    dist.barrier()
    return checkpoint_dir


def _rank_training_state_path(checkpoint_dir: str, global_rank: int) -> str:
    return os.path.join(
        checkpoint_dir,
        f"training_state.rank{int(global_rank)}.pt",
    )


def _serialize_training_state(
    *,
    optimizer,
    next_micro_step: int,
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
):
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    return {
        "next_micro_step": int(next_micro_step),
        "optimizer": optimizer.state_dict(),
        "global_rank": int(global_rank),
        "world_size": int(world_size),
        "local_batch_size": int(local_batch_size),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def _full_model_state_dict(model):
    assert isinstance(model, FSDPModule), "training model must be wrapped in FSDP2"
    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def _save_model_checkpoint(*, model, draft_model, checkpoint_dir: str):
    state_dict = _full_model_state_dict(model)
    if is_global_main_process():
        draft_state_dict = {}
        for key, value in state_dict.items():
            normalized_key = key
            if normalized_key.startswith("_orig_mod."):
                normalized_key = normalized_key[len("_orig_mod.") :]
            draft_state_dict[normalized_key] = value
        assert draft_state_dict, "Failed to extract draft model state_dict from checkpoint."
        draft_model.save_pretrained(checkpoint_dir, state_dict=draft_state_dict)
