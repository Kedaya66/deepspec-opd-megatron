"""统一的 target 特征服务客户端。

2026-08-13 重构动机(两天基准实验的直接产物):
  - 此前 rollout / 缓存重放 / teacher-forced 三条请求路径的重试逻辑分散在
    opd_rollout_loader.py 与 dspark_online_trainer.py 各处,三次事故分别补丁:
    (a) 缓存重放裸抛 ConnectionError 打崩训练;
    (b) teacher-forced 回退裸抛 + CUDAPrefetcher 后台线程吞异常 => 数据流静默
        提前结束 -> 收尾 checkpoint 断言二次崩;
    (c) 600s 短超时在服务重负载下引发"放弃-重算"恶性循环(服务端算完的
        请求客户端已放弃,吞吐浪费近半)。
  - 统一到一个客户端类:一处重试策略、一处编解码、一处端点轮询。
    loader 与 trainer 都从这里取数,消除 loader -> trainer 的反向依赖。

重试哲学(与服务自愈哨兵配合,见 scripts/ops/svc_sentinel.sh):
  - 客户端不负责判断服务死活(哨兵负责),只负责"等到成功或耗尽重试"。
  - 超时取 1200s:比正常请求(60~90s)宽一个数量级;短超时会造成放弃-重算循环。
  - 重试间隔线性升到 30s 封顶;120 次约可扛 1 小时服务重启窗口。
"""

import base64
import itertools
import threading
import time

import numpy as np
import torch


def decode_hidden(hs):
    """解 b64 分段 hidden,统一返回 bfloat16。

    bf16 段直接 view,不经过 fp32 往返(实测省 8% CPU、峰值内存减半)。
    dtype 全段一致,否则 torch.cat 混 dtype 报错。
    """
    if isinstance(hs, list) and hs and isinstance(hs[0], str):
        segs = []
        for seg in hs:
            if isinstance(seg, str) and seg.startswith("shm:"):
                # 共享内存直通段(服务端 SGLANG_DSPARK_EXPORT_SHM=1):
                # "shm:tag:rows:cols:path"。同机 tmpfs 直读,读后即删。
                import os

                _, tag, r, c, path = seg.split(":", 4)
                r, c = int(r), int(c)
                if tag == "bf16":
                    arr = np.fromfile(path, dtype=np.uint16)
                    t = torch.from_numpy(arr).view(torch.bfloat16).reshape(r, c)
                else:
                    t = torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(r, c).to(torch.bfloat16)
                try:
                    os.unlink(path)
                except OSError:
                    pass
                segs.append(t)
                continue
            if isinstance(seg, str):
                tag, r, c, b64 = seg.split(":", 3)
                raw = base64.b64decode(b64)
                r, c = int(r), int(c)
                if tag == "bf16":
                    arr = np.frombuffer(raw, dtype=np.uint16).copy()
                    segs.append(torch.from_numpy(arr).view(torch.bfloat16).reshape(r, c))
                else:
                    arr = np.frombuffer(raw, dtype=np.float32).copy()
                    segs.append(torch.from_numpy(arr).reshape(r, c).to(torch.bfloat16))
            else:  # 慢路 decode 行(list of float)
                segs.append(
                    torch.as_tensor(seg, dtype=torch.float32).reshape(1, -1).to(torch.bfloat16)
                )
        return segs[0] if len(segs) == 1 else torch.cat(segs, 0)
    t = torch.as_tensor(hs, dtype=torch.float32).to(torch.bfloat16)
    return t.squeeze(0) if t.dim() == 3 else t


class FeatureClient:
    """target 特征服务的唯一入口:rollout / 重放 / teacher-forced prefill。

    线程安全:可被 loader 的线程池并发使用。
    """

    def __init__(self, endpoints, *, timeout=1200.0, max_retries=120, tag="dspark-opd"):
        if isinstance(endpoints, str):
            endpoints = [endpoints]
        self.endpoints = [
            e if str(e).startswith("http") else f"http://{e}" for e in endpoints
        ]
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.tag = tag
        self._rr = itertools.count()
        self._lock = threading.Lock()

    def _endpoint(self):
        return self.endpoints[next(self._rr) % len(self.endpoints)]

    # ---------- 底层:带重试的 /generate ----------
    def _generate(self, payload):
        import requests

        endpoint = self._endpoint()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{endpoint}/generate", json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
                ValueError,  # resp.json() 解析失败(截断的响应)
            ) as e:
                last_err = e
                wait = min(30.0, 5.0 * (attempt + 1))
                print(
                    f"[{self.tag}] feature service retry "
                    f"{attempt + 1}/{self.max_retries} in {wait:.0f}s "
                    f"({type(e).__name__}; {endpoint})",
                    flush=True,
                )
                time.sleep(wait)
                endpoint = self._endpoint()  # 换端点重试(多实例时)
        raise RuntimeError(
            f"feature service unavailable after {self.max_retries} retries"
        ) from last_err

    # ---------- 三种业务请求 ----------
    def rollout(self, prompt_ids, *, max_new_tokens, temperature=0.0, top_p=1.0):
        """target 自回归 rollout,返回 (gen_ids, packed[L+gen-1, (n+1)*H])。

        末生成 token 未被前向处理,无 hidden 行 —— 调用方按此对齐 loss 区。
        """
        d = self._generate({
            "input_ids": list(prompt_ids),
            "sampling_params": {
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
            "return_hidden_states": True,
        })
        gen_ids = list(d.get("output_ids") or [])
        packed = decode_hidden(d["meta_info"]["hidden_states"])
        return gen_ids, packed

    def prefill_features(self, ids, *, rows=None):
        """一次 prefill 取全序列特征(teacher-forced / 缓存重放共用)。

        rows: 返回的行数上限;缓存重放传 len(ids)-1 对齐 rollout 口径,
              teacher-forced 传 len(ids)。
        """
        d = self._generate({
            "input_ids": list(ids),
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_hidden_states": True,
        })
        pk = decode_hidden(d["meta_info"]["hidden_states"])
        return pk[: rows if rows is not None else len(ids)]
