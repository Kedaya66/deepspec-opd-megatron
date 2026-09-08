"""SGLang 版 target 特征提取（解耦：服务独立部署，本函数只做 HTTP 客户端）。

drop-in 替代 run_target_forward_with_hooks：输入/输出数据结构不变，
仅把 target_model 换成已部署服务的 endpoint，业务逻辑改为调 native /generate。

服务侧（独立启动，见 run_sglang_dspark_export.sh）需带：
    SGLANG_DSPARK_EXPORT_LAYERS="40,41,42"
    --moe-runner-backend marlin
    --enable-return-hidden-states
    --disable-cuda-graph
    --disable-radix-cache
此时 native POST /generate 的 meta_info["hidden_states"] 逐 prompt token 返回
    cat([L40, L41, L42, last_norm])  形状 [seq, (N+1)*H]
前 N 块 == target_hidden_states，末块 == target_last_hidden_states。
注意：必须走 native /generate（返回全 token）；OpenAI /v1/completions 只返回最后一个 token。
"""
from dataclasses import dataclass

import base64

import numpy as np
import requests
import torch


def _decode_hidden(hs):
    """解码 meta_info["hidden_states"]：兼容 base64 快路 与 旧的嵌套 float list。"""
    if isinstance(hs, list) and hs and isinstance(hs[0], str):
        tag, r, c, b64 = hs[0].split(":", 3)
        raw = base64.b64decode(b64)
        r, c = int(r), int(c)
        if tag == "bf16":
            arr = np.frombuffer(raw, dtype=np.uint16).copy()
            return torch.from_numpy(arr).view(torch.bfloat16).reshape(r, c).float()
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        return torch.from_numpy(arr).reshape(r, c)
    t = torch.as_tensor(hs, dtype=torch.float32)
    return t.squeeze(0) if t.dim() == 3 else t


@dataclass(frozen=True)
class TargetForwardResult:
    target_hidden_states: torch.Tensor        # [B, T, N*H]
    target_last_hidden_states: torch.Tensor   # [B, T, H]


def run_target_forward_via_sglang(
    *,
    endpoint: str,                 # 已部署服务地址，如 "http://127.0.0.1:30000"
    input_ids: torch.Tensor,       # [B, T]
    attention_mask: torch.Tensor,  # [B, T]
    target_layer_ids,              # 如 [40, 41, 42]
    timeout: float = 1200.0,
    session: "requests.Session | None" = None,
):
    """与 run_target_forward_with_hooks 同签名同返回，仅 target_model -> endpoint。

    attention_mask 只在客户端用于去 padding / 取真实长度；SGLang 用 ragged 布局
    （无 padding），故不把 mask 传给服务。
    """
    n_layers = len([int(x) for x in target_layer_ids])
    B, T = input_ids.shape
    post = (session or requests).post

    th_list, tl_list = [], []
    for b in range(B):
        mask_b = attention_mask[b].bool()
        ids = input_ids[b][mask_b].tolist()   # 去 padding，只送真实 token
        L = len(ids)

        resp = post(
            f"{endpoint}/generate",
            json={
                "input_ids": ids,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                "return_hidden_states": True,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        hs = resp.json()["meta_info"]["hidden_states"]
        packed = _decode_hidden(hs)             # base64 快路解码（或兼容旧 list）
        packed = packed[:L]                     # 只取 prompt token

        H = packed.shape[-1] // (n_layers + 1)  # 由返回维度反推 hidden_size
        th = packed[:, : n_layers * H]          # 前 N 块 == target_hidden_states
        tl = packed[:, n_layers * H:]           # 末块     == target_last_hidden_states

        # 回填到 [T, ·]（右 padding，与原主循环 [b, :seq_len] 取法一致）
        th_full = torch.zeros(T, th.shape[-1], dtype=torch.bfloat16)
        tl_full = torch.zeros(T, tl.shape[-1], dtype=torch.bfloat16)
        th_full[:L] = th.to(torch.bfloat16)
        tl_full[:L] = tl.to(torch.bfloat16)
        th_list.append(th_full)
        tl_list.append(tl_full)

    return TargetForwardResult(
        target_hidden_states=torch.stack(th_list, dim=0),       # [B, T, N*H]
        target_last_hidden_states=torch.stack(tl_list, dim=0),  # [B, T, H]
    )
