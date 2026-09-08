"""OPD rollout 结果缓存:只存 target 生成的 token ids,不存特征。

动机(2026-08-11 实测):
  - 服务侧是瓶颈。4 卡 SGLang 跑 DeepSeek-V4-Flash、prompt 25k,总吞吐约
    1930 tok/s,总并发 8-16 即饱和(1212 -> 1519 tok/s),再加并发只丢连接。
    因此客户端并发/编码都没有优化空间。
  - 但 prompt 25000 token 对生成 512 token 来说,98% 的 token 属 prefill。
    rollout 走 512 步串行 decode(实测 13.2 s/样本),而 teacher-forced 只需
    一次 prefill(实测 4.4 s/样本)。
  - target 模型是冻结的,rollout 由 target 产生 => 给定 prompt,生成结果是
    合法的 on-policy 样本,与何时生成无关。故可缓存复用:首轮付 decode 的钱,
    之后所有 epoch / 所有实验都改走 prefill。
  端到端实测(2026-08-11):gbs=64/128/512 提速 2.90/3.02/2.95x,命中率 94-95%,
  loss 距同配置 5 轮均值 -0.6σ(统计上不可区分)。

  只缓存 gen_ids(每样本约 2 KB,全数据集约 11 MB);特征不缓存 —— 按 25k 序列
  每样本 819 MB,5632 样本要 4.6 TB,磁盘只有 475 GB。

并发:每个 rank 一个 loader、多线程写,故用 tmp+rename 原子落盘,天然幂等。
"""

import hashlib
import os
import struct
import threading


CACHE_VERSION = 1


class RolloutCache:
    """prompt -> gen_ids 的磁盘缓存。key 含采样参数与权重版本,避免串味。"""

    def __init__(self, cache_dir, *, max_new_tokens, temperature, top_p,
                 weight_version="default", max_length=None):
        self.dir = os.path.abspath(cache_dir)
        os.makedirs(self.dir, exist_ok=True)
        # 采样参数变了 / 换权重了,缓存必须失效。max_length 也要进 key:
        # budget = min(max_new, max_length - prompt_len),max_length 变则同一
        # prompt 的合法生成长度变,复用旧结果是静默的语义漂移。
        self._salt = (
            f"v{CACHE_VERSION}|mnt={int(max_new_tokens)}|t={float(temperature):.6g}"
            f"|p={float(top_p):.6g}|w={weight_version}"
            + (f"|L={int(max_length)}" if max_length is not None else "")
        ).encode()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, prompt_ids):
        h = hashlib.sha256(self._salt)
        # prompt 以 int32 小端入哈希,避免 str() 的歧义与开销
        h.update(struct.pack(f"<{len(prompt_ids)}i", *prompt_ids))
        return h.hexdigest()

    def _path(self, key):
        # 两级目录,避免单目录几万文件
        return os.path.join(self.dir, key[:2], key[2:] + ".ids")

    def get(self, prompt_ids):
        """命中返回 list[int];未命中返回 None。"""
        p = self._path(self._key(prompt_ids))
        try:
            with open(p, "rb") as f:
                raw = f.read()
        except (FileNotFoundError, NotADirectoryError):
            with self._lock:
                self.misses += 1
            return None
        if len(raw) % 4 != 0:  # 截断的坏文件:当未命中
            with self._lock:
                self.misses += 1
            return None
        n = len(raw) // 4
        with self._lock:
            self.hits += 1
        return list(struct.unpack(f"<{n}i", raw))

    def put(self, prompt_ids, gen_ids):
        if not gen_ids:
            return
        p = self._path(self._key(prompt_ids))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(struct.pack(f"<{len(gen_ids)}i", *[int(x) for x in gen_ids]))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)  # 原子;多 rank 同 key 互相覆盖也无害
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def stats(self):
        with self._lock:
            h, m = self.hits, self.misses
        tot = h + m
        return h, m, (100.0 * h / tot if tot else 0.0)
