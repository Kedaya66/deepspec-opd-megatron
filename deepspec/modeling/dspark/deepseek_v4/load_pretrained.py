"""Load pretrained DeepSeek-V4 DSpark (Flash/Pro) weights into a DeepSpec draft.

The released ``DeepSeek-V4-{Flash,Pro}-DSpark`` checkpoints store the DSpark module
under ``mtp.*`` with FP8/FP4 quantized weights. This module reads those tensors,
dequantizes them to bfloat16, renames them to the DeepSpec draft layout, and copies
whatever is shape-compatible into an already-built ``DeepSeekV4*DSparkModel`` (the
rest of the draft keeps its random init).

Only the serving-only tensors with no home in the dense trainable draft are skipped:
the Hyper-Connection mixers (``hc_*``) and the attention sink (``attn.attn_sink``).

Two entry points:
  * ``apply_pretrained_dspark_weights(model, ckpt_path)`` — copy compatible weights
    into an existing model (used by the trainers to optionally warm-start).
  * ``build_pretrained_draft_model(ckpt_path, variant)`` — build a draft whose config
    matches the checkpoint and fully load it (used by the ``load_weights`` CLI).
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import torch
from safetensors import safe_open
from transformers import AutoConfig, PretrainedConfig


# FP4 (e2m1) code -> value table, matching the reference inference/convert.py.
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

FP8_BLOCK = 128
FP4_BLOCK = 32


class _AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------
def _e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    exps = scale.view(torch.uint8).to(torch.float32)
    return torch.pow(torch.tensor(2.0), exps - 127.0)


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    """float8_e4m3fn weight with a per-(block x block) e8m0 scale -> bfloat16."""
    w = weight.to(torch.float32)
    out_dim, in_dim = w.shape
    s = _e8m0_to_float(scale)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:out_dim, :in_dim]
    return (w * s).to(torch.bfloat16)


def dequant_fp4(weight_i8: torch.Tensor, scale: torch.Tensor, block: int = FP4_BLOCK) -> torch.Tensor:
    """int8-packed e2m1 weight (2 fp4/byte, low nibble first) + per-row 32-block e8m0 scale -> bf16."""
    table = FP4_TABLE.to(weight_i8.device)
    x = weight_i8.view(torch.uint8)
    low = (x & 0x0F).long()
    high = ((x >> 4) & 0x0F).long()
    vals = torch.stack([table[low], table[high]], dim=-1).flatten(1)
    s = _e8m0_to_float(scale).repeat_interleave(block, dim=1)[:, : vals.shape[1]]
    return (vals * s).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Key remapping: reference (serving) layout -> DeepSpec module layout
# ---------------------------------------------------------------------------
def remap_key(name: str) -> str | None:
    """Map a checkpoint parameter name to its DeepSpec name, or None to drop it."""
    if name in ("embed.weight", "model.embed.weight"):
        return "embed_tokens.weight"
    if name in ("head.weight", "model.head.weight"):
        return "lm_head.weight"

    if name.startswith("model."):
        name = name[len("model."):]
    if not name.startswith("mtp."):
        return None

    parts = name.split(".")
    stage = parts[1]
    rest = parts[2:]
    tail = ".".join(rest)

    if rest[0].startswith("hc_") or tail.startswith("attn.attn_sink"):
        return None  # Hyper-Connection mixers / attention sink: no dense counterpart.

    if rest[0] == "attn":
        return f"layers.{stage}.self_attn." + ".".join(rest[1:])
    if rest[0] == "attn_norm":
        return f"layers.{stage}.input_layernorm.{rest[-1]}"
    if rest[0] == "ffn_norm":
        return f"layers.{stage}.post_attention_layernorm.{rest[-1]}"
    if rest[0] == "ffn":
        if rest[1] == "gate":
            if rest[2] == "weight":
                return f"layers.{stage}.mlp.gate.router.weight"
            if rest[2] == "bias":
                return f"layers.{stage}.mlp.gate.bias"
            return None
        expert_w = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        if rest[1] == "experts":
            return f"layers.{stage}.mlp.experts.{rest[2]}.{expert_w[rest[3]]}.weight"
        if rest[1] == "shared_experts":
            return f"layers.{stage}.mlp.shared_expert.{expert_w[rest[2]]}.weight"
        return None
    if rest[0] == "main_proj":
        return f"fc.{rest[-1]}"
    if rest[0] == "main_norm":
        return "hidden_norm.weight"
    if rest[0] == "norm":
        return "norm.weight"
    if rest[0] == "markov_head":
        return "markov_head." + ".".join(rest[1:])
    if rest[0] == "confidence_head":
        return "confidence_head." + ".".join(rest[1:])
    return None


# ---------------------------------------------------------------------------
# Checkpoint inspection
# ---------------------------------------------------------------------------
def load_index(ckpt_path: str) -> dict:
    return json.load(open(os.path.join(ckpt_path, "model.safetensors.index.json")))["weight_map"]


def load_target_config(ckpt_path: str) -> PretrainedConfig:
    # Use the registered model config class (handles RoPE standardization etc.);
    # falls back to a raw dict-backed config only if the model_type is unknown.
    try:
        return AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    except Exception:
        cfg_dict = json.load(open(os.path.join(ckpt_path, "config.json")))
        config = PretrainedConfig.from_dict(cfg_dict)
        config = config[0] if isinstance(config, tuple) else config
        config.model_type = cfg_dict.get("model_type", "deepseek_v4")
        return config


def detect_variant(config) -> str:
    hidden = int(getattr(config, "hidden_size", 0))
    if hidden == 4096:
        return "flash"
    if hidden == 7168:
        return "pro"
    name = (
        " ".join(getattr(config, "architectures", []) or []).lower()
        + " "
        + str(getattr(config, "_name_or_path", "")).lower()
    )
    if "pro" in name:
        return "pro"
    if "flash" in name:
        return "flash"
    raise ValueError(f"Cannot auto-detect variant from hidden_size={hidden}; pass variant explicitly.")


def detect_num_stages(weight_map: dict) -> int:
    stages = {int(k.split(".")[1]) for k in weight_map if k.startswith("mtp.")}
    assert stages == set(range(len(stages))), f"Non-contiguous mtp stages: {sorted(stages)}"
    return len(stages)


def assert_is_dspark_checkpoint(weight_map: dict, ckpt_path: str) -> None:
    if not any("markov_head" in k for k in weight_map):
        mtp_like = sorted({re.sub(r"\.\d+\.", ".{N}.", k) for k in weight_map if k.startswith("mtp.")})
        raise ValueError(
            f"{ckpt_path} is not a DSpark checkpoint (no mtp.*.markov_head); it looks like a "
            f"base V4 model with a vanilla MTP head. Found mtp patterns: {mtp_like[:12]}"
        )


# ---------------------------------------------------------------------------
# State-dict construction and application
# ---------------------------------------------------------------------------
def _wanted(name: str) -> bool:
    base = name[len("model."):] if name.startswith("model.") else name
    return base.startswith("mtp.") or base in ("embed.weight", "head.weight")


def build_dspark_state_dict(ckpt_path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Read mtp.*/embed/head tensors, dequantize, return them under DeepSpec keys."""
    weight_map = load_index(ckpt_path)
    assert_is_dspark_checkpoint(weight_map, ckpt_path)

    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        if _wanted(key):
            shard_to_keys[shard].append(key)

    raw: dict[str, torch.Tensor] = {}
    for shard, keys in shard_to_keys.items():
        with safe_open(os.path.join(ckpt_path, shard), framework="pt", device=device) as f:
            for key in keys:
                raw[key] = f.get_tensor(key)

    out: dict[str, torch.Tensor] = {}
    n_fp8 = n_fp4 = n_plain = n_dropped = 0
    for key, tensor in raw.items():
        if key.endswith(".scale"):
            continue
        dst = remap_key(key)
        if dst is None:
            n_dropped += 1
            continue
        scale_key = key[: -len(".weight")] + ".scale" if key.endswith(".weight") else None
        scale = raw.get(scale_key) if scale_key else None
        if scale is not None:
            value = dequant_fp4(tensor, scale) if tensor.dtype == torch.int8 else dequant_fp8_block(tensor, scale)
            if tensor.dtype == torch.int8:
                n_fp4 += 1
            else:
                n_fp8 += 1
        else:
            value = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            n_plain += 1
        out[dst] = value

    print(
        f"[dspark-load] dequant fp8={n_fp8} fp4={n_fp4} | copied plain={n_plain} | "
        f"dropped(no destination)={n_dropped} | mapped tensors={len(out)}"
    )
    return out


def apply_pretrained_dspark_weights(model, ckpt_path: str, device: str = "cpu") -> dict:
    """Copy shape-compatible pretrained DSpark weights into ``model`` (strict=False).

    Params whose name/shape does not match (e.g. a slimmer draft with fewer experts
    or layers, or the dropped HC/sink tensors) are left at their current init and
    reported. Returns a summary dict.
    """
    state_dict = build_dspark_state_dict(ckpt_path, device=device)
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    feed, mismatch = {}, []
    for key, value in state_dict.items():
        if key not in model_keys:
            mismatch.append((key, "no such param in model", tuple(value.shape)))
            continue
        expected = tuple(model_sd[key].shape)
        if tuple(value.shape) != expected:
            mismatch.append((key, expected, tuple(value.shape)))
            continue
        feed[key] = value.to(dtype=model_sd[key].dtype)

    result = model.load_state_dict(feed, strict=False)
    covered = sorted(model_keys & set(feed.keys()))
    report = {
        "num_model_params": len(model_keys),
        "num_loaded": len(covered),
        "missing_keys": list(result.missing_keys),
        "mismatch": mismatch,
    }
    print(
        f"[dspark-load] params in model={report['num_model_params']} | "
        f"loaded={report['num_loaded']} | still-random={len(report['missing_keys'])} | "
        f"mismatched/absent-in-model={len(mismatch)}"
    )
    if mismatch:
        print("[dspark-load] WARNING: mismatches (first 10):")
        for row in mismatch[:10]:
            print("   ", row)
    if result.missing_keys:
        print(f"[dspark-load] NOTE: {len(result.missing_keys)} params kept at init (first 20):")
        for k in result.missing_keys[:20]:
            print("   ", k)
    return report


def load_embed_and_head(ckpt_path: str, device: str = "cpu"):
    """Read just ``embed.weight`` and ``head.weight`` (input/output embeddings) from a
    checkpoint, without loading the full (possibly hundreds-of-billions-param) target.

    Returns (embed_weight, lm_head_weight) as bf16 tensors on ``device``.
    """
    weight_map = load_index(ckpt_path)

    def _find(*candidates):
        for c in candidates:
            if c in weight_map:
                return c
        raise KeyError(f"none of {candidates} in checkpoint index at {ckpt_path}")

    embed_key = _find("embed.weight", "model.embed.weight", "model.embed_tokens.weight")
    head_key = _find("head.weight", "lm_head.weight", "model.head.weight")
    out = {}
    for logical, key in (("embed", embed_key), ("head", head_key)):
        shard = weight_map[key]
        with safe_open(os.path.join(ckpt_path, shard), framework="pt", device=device) as f:
            out[logical] = f.get_tensor(key).to(torch.bfloat16)
    return out["embed"], out["head"]


def build_pretrained_draft_model(ckpt_path: str, variant: str | None = None):
    """Build a DeepSpec draft whose config matches the checkpoint and fully load it.

    Returns (model, variant).
    """
    # Local imports to avoid any import cycle with the package __init__.
    from deepspec.modeling.dspark.deepseek_v4.config import (
        build_flash_draft_config,
        build_pro_draft_config,
    )
    from deepspec.modeling.dspark.deepseek_v4.modeling import (
        DeepSeekV4FlashDSparkModel,
        DeepSeekV4ProDSparkModel,
    )

    variants = {
        "flash": (DeepSeekV4FlashDSparkModel, build_flash_draft_config),
        "pro": (DeepSeekV4ProDSparkModel, build_pro_draft_config),
    }
    weight_map = load_index(ckpt_path)
    assert_is_dspark_checkpoint(weight_map, ckpt_path)
    target_config = load_target_config(ckpt_path)
    variant = variant or detect_variant(target_config)
    assert variant in variants, f"variant must be one of {list(variants)}, got {variant!r}"
    model_cls, build_cfg = variants[variant]

    num_stages = detect_num_stages(weight_map)
    has_confidence = any("confidence_head" in k for k in weight_map)

    def _cfg(name, default):
        value = getattr(target_config, name, None)
        return default if value is None else value

    model_args = _AttrDict(
        num_draft_layers=num_stages,
        target_layer_ids=list(_cfg("dspark_target_layer_ids", [40, 41, 42])),
        block_size=int(_cfg("dspark_block_size", 5)),
        mask_token_id=int(_cfg("dspark_noise_token_id", 128799)),
        num_anchors=512,
        markov_rank=int(_cfg("dspark_markov_rank", 256)),
        markov_head_type="vanilla",
        confidence_head_alpha=1.0 if has_confidence else 0.0,
        confidence_head_with_markov=True,
    )
    draft_config = build_cfg(target_config=target_config, model_args=model_args)
    draft_config._attn_implementation = "eager"
    if getattr(draft_config, "quantization_config", None) is not None:
        draft_config.quantization_config = None
    print(
        f"[dspark-load] variant={variant} stages={num_stages} "
        f"n_routed_experts={getattr(draft_config, 'n_routed_experts', '?')} "
        f"markov_rank={model_args['markov_rank']} confidence={has_confidence}"
    )
    return model_cls(draft_config), variant


__all__ = [
    "apply_pretrained_dspark_weights",
    "build_pretrained_draft_model",
    "load_embed_and_head",
    "build_dspark_state_dict",
    "remap_key",
    "detect_variant",
    "detect_num_stages",
    "assert_is_dspark_checkpoint",
    "dequant_fp8_block",
    "dequant_fp4",
]
