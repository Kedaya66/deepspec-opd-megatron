"""Load official DSpark ``mtp.*`` weights into a match-v1 draft.

On top of the base loader (dequant + rename) this:
  * keeps ``attn.attn_sink`` and all ``hc_*`` tensors (24 + hyper-head), mapping
    them to the match-v1 modules instead of dropping them;
  * applies the rope-channel permutation (interleaved -> rotate_half layout) to
    the four weight groups whose channels the rope convention touches:
    ``wq_b`` per-head trailing rows, ``wkv`` trailing rows, ``kv_norm`` trailing
    entries, and ``wo_a`` per-head trailing input columns.

The permutation is applied EXACTLY ONCE, here, when converting from the
official layout. Checkpoints saved from a match-v1 model are already in the
permuted layout and must NOT go through this module again.
"""

from __future__ import annotations

import torch

from deepspec.modeling.dspark.deepseek_v4.load_pretrained import (
    build_dspark_state_dict as _base_build,
)
from deepspec.modeling.dspark.deepseek_v4 import load_pretrained as _base

# 捕获未被 monkeypatch 的原始 remap_key,供兜底使用(否则 build_match_v1_state_dict
# 打补丁后,下面的兜底会调到自己 -> 无限递归)。
_ORIG_REMAP = _base.remap_key


def _remap_key_match_v1(name: str) -> str | None:
    base = name[len("model."):] if name.startswith("model.") else name
    if base.startswith("mtp."):
        parts = base.split(".")
        stage, rest = parts[1], ".".join(parts[2:])
        if rest == "attn.attn_sink":
            return f"layers.{stage}.self_attn.attn_sink"
        for site, module in (("hc_attn", "attn_hc"), ("hc_ffn", "ffn_hc")):
            for leaf in ("fn", "base", "scale"):
                if rest == f"{site}_{leaf}":
                    return f"layers.{stage}.{module}.{leaf}"
        for leaf in ("fn", "base", "scale"):
            if rest == f"hc_head_{leaf}":
                return f"hc_head.{leaf}"
    return _ORIG_REMAP(name)


ROPE_PERM = torch.cat([torch.arange(0, 64, 2), torch.arange(1, 64, 2)])


def permute_rope_channels_(state_dict: dict, *, num_heads: int = 64, head_dim: int = 512,
                           rope_dim: int = 64, o_groups: int = 8) -> int:
    """In-place interleaved -> rotate_half channel permutation. Returns #tensors touched."""
    perm = ROPE_PERM
    nope = head_dim - rope_dim
    heads_per_group = num_heads // o_groups
    touched = 0
    for key, t in state_dict.items():
        if key.endswith("self_attn.wq_b.weight"):
            w = t.view(num_heads, head_dim, -1)
            w[:, nope:] = w[:, nope:][:, perm]
        elif key.endswith("self_attn.wkv.weight"):
            t[nope:] = t[nope:][perm]
        elif key.endswith("self_attn.kv_norm.weight"):
            t[nope:] = t[nope:][perm]
        elif key.endswith("self_attn.wo_a.weight"):
            w = t.view(t.shape[0], heads_per_group, head_dim)
            w[..., nope:] = w[..., nope:][..., perm]
        else:
            continue
        touched += 1
    return touched


def build_match_v1_state_dict(ckpt_path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Official layout -> match-v1 layout (rename + dequant + rope permutation)."""
    original_remap = _base.remap_key
    _base.remap_key = _remap_key_match_v1
    try:
        state_dict = _base_build(ckpt_path, device=device)
    finally:
        _base.remap_key = original_remap
    touched = permute_rope_channels_(state_dict)
    print(f"[match-v1 load] rope-permuted tensors: {touched} (expect 4 per layer)")
    return state_dict


def apply_pretrained_match_v1_weights(model, ckpt_path: str, device: str = "cpu") -> dict:
    """Copy shape-compatible official weights into a match-v1 model (strict=False)."""
    state_dict = build_match_v1_state_dict(ckpt_path, device=device)
    model_sd = model.state_dict()
    feed, mismatch = {}, []
    for key, value in state_dict.items():
        if key not in model_sd:
            mismatch.append((key, "no such param in model", tuple(value.shape)))
            continue
        if tuple(value.shape) != tuple(model_sd[key].shape):
            mismatch.append((key, tuple(model_sd[key].shape), tuple(value.shape)))
            continue
        feed[key] = value.to(dtype=model_sd[key].dtype)
    result = model.load_state_dict(feed, strict=False)
    report = {
        "num_model_params": len(model_sd),
        "num_loaded": len(feed),
        "missing_keys": list(result.missing_keys),
        "mismatch": mismatch,
    }
    print(
        f"[match-v1 load] params in model={report['num_model_params']} | "
        f"loaded={report['num_loaded']} | still-random={len(report['missing_keys'])} | "
        f"mismatched/absent-in-model={len(mismatch)}"
    )
    for row in mismatch[:10]:
        print("   MISMATCH:", row)
    for k in report["missing_keys"][:10]:
        print("   still-random:", k)
    return report


__all__ = [
    "build_match_v1_state_dict",
    "apply_pretrained_match_v1_weights",
    "permute_rope_channels_",
    "ROPE_PERM",
]
