"""OPD 并发 rollout 预取 loader(乱序消费 + 自适应窗口 + gen_ids 缓存)。

包在 DataLoader 外:read-ahead 窗口内的 batch 并发 rollout(吃 SGLang
continuous batching 聚合吞吐);**谁先完成先消费**(SGD 下窗口内样本可交换,
消除队头 straggler 阻塞);按"训练端等待时长"反馈调窗;多实例按样本轮询分流。
yield 纯张量 batch(input_ids/attention_mask/loss_mask/opd_th/opd_tl),下游零 HTTP。

2026-08-13 重构:全部 HTTP 与重试收敛到 FeatureClient(一处策略);
三条取数路径(缓存重放 / rollout / teacher-forced 回退)共用 _pack 组装,
产出逐位一致(有 mock 对拍测试)。

单条 rollout 生成过短(≤1 token)自动回退 teacher-forced,保证训练不中断。
"""
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import torch

from deepspec.data.feature_client import FeatureClient


class RolloutPrefetchLoader:
    def __init__(
        self,
        dataloader,
        *,
        endpoints,
        n_target_layers,
        max_length,
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
        timeout=1200.0,
        prefetch=4,
        max_prefetch=8,
        min_prefetch=2,
        cache_dir=None,
        weight_version="default",
    ):
        self.dataloader = dataloader
        self.client = FeatureClient(endpoints, timeout=timeout)
        # rank 前缀:4 个 rank 混打同一 stdout,无标识时无法归属"谁在等"
        # (2026-08-13 排障实证:死锁期无法从日志判定饿死的 rank)。
        # 注意用 dist.get_rank():本项目 mp.spawn 启动,LOCAL_RANK 环境变量不存在。
        try:
            import torch.distributed as _dist

            _r = _dist.get_rank() if _dist.is_available() and _dist.is_initialized() else "-"
        except Exception:
            _r = "-"
        self._tag = f"[opd-prefetch r{_r}]"
        self.n_layers = int(n_target_layers)
        self.max_length = int(max_length)
        self.max_new = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.window = max(1, int(prefetch))
        self.max_window = max(self.window, int(max_prefetch))
        self.min_window = max(1, int(min_prefetch))
        self.executor = ThreadPoolExecutor(max_workers=self.max_window)
        # gen_ids 缓存(可选)。target 冻结 => 缓存的生成是合法 on-policy 样本,
        # 命中后用一次 prefill 重建特征替代串行 decode。
        # 0731 权重实测 gbs=64:27.3 -> 5.3 min(5.2x),loss 差距 < 1σ。
        self.cache = None
        if cache_dir:
            from deepspec.data.rollout_cache import RolloutCache

            self.cache = RolloutCache(
                cache_dir,
                max_new_tokens=self.max_new,
                temperature=self.temperature,
                top_p=self.top_p,
                # budget = min(max_new, max_length - prompt_len) 依赖 max_length;
                # 换权重必须失效 —— 两者都进 key
                max_length=self.max_length,
                weight_version=weight_version,
            )

    def __len__(self):
        return len(self.dataloader)

    def _pack(self, prompt, gen_ids, packed, ids, lm, prompt_len):
        """由 (prompt, gen_ids, packed) 组装 (seq, loss_mask, th, tl)。

        三条取数路径共用,保证产出逐位一致。packed 行数 = len(seq)-1
        (末 token 未被前向处理,无 hidden 行,不进 loss 区)。
        """
        seq = torch.tensor(prompt + list(gen_ids), dtype=ids.dtype)
        H = packed.shape[-1] // (self.n_layers + 1)
        rows = min(int(packed.shape[0]), int(seq.shape[0]))
        th = torch.zeros(seq.shape[0], self.n_layers * H, dtype=torch.bfloat16)
        tl = torch.zeros(seq.shape[0], H, dtype=torch.bfloat16)
        th[:rows] = packed[:rows, : self.n_layers * H].to(torch.bfloat16)
        tl[:rows] = packed[:rows, self.n_layers * H :].to(torch.bfloat16)
        new_lm = torch.zeros(seq.shape[0], dtype=lm.dtype)
        hi = min(int(seq.shape[0]), rows)
        if hi > prompt_len:
            new_lm[prompt_len:hi] = 1
        return seq, new_lm, th, tl

    # ---------- per-sample worker(线程池内跑,含 HTTP)----------
    def _rollout_one(self, ids, lm):
        """ids/lm: 1D cpu tensor(去 padding)。返回 (seq_ids, loss_mask, th, tl)。"""
        nz = torch.nonzero(lm, as_tuple=False)
        prompt_len = int(nz[0]) if len(nz) else int(ids.shape[0])
        prompt = ids[:prompt_len].tolist()
        budget = max(1, min(self.max_new, self.max_length - prompt_len))

        # ---- 路径 1:缓存命中,一次 prefill 重建特征 ----
        if self.cache is not None:
            hit = self.cache.get(prompt)
            if hit is not None:
                hit = hit[:budget]
                if len(hit) >= 2:
                    seq_list = prompt + hit
                    # FeatureClient 内置重试;真到重试耗尽会 raise(可见地崩),
                    # 不再需要"命中失败回退 rollout"—— rollout 走的是同一个
                    # 客户端同一个服务,它也活不了。
                    pk = self.client.prefill_features(seq_list, rows=len(seq_list) - 1)
                    return self._pack(prompt, hit, pk, ids, lm, prompt_len)

        # ---- 路径 2:真 rollout ----
        gen_ids, packed = self.client.rollout(
            prompt,
            max_new_tokens=budget,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        gen_ids = gen_ids[:budget]
        if len(gen_ids) >= 2:
            if self.cache is not None:
                self.cache.put(prompt, gen_ids)
            return self._pack(prompt, gen_ids, packed, ids, lm, prompt_len)

        # ---- 路径 3:生成过短,teacher-forced 原文本 + prefill 特征 ----
        full = ids.tolist()
        packed = self.client.prefill_features(full, rows=len(full))
        H = packed.shape[-1] // (self.n_layers + 1)
        th = torch.zeros(ids.shape[0], self.n_layers * H, dtype=torch.bfloat16)
        tl = torch.zeros(ids.shape[0], H, dtype=torch.bfloat16)
        th[: packed.shape[0]] = packed[:, : self.n_layers * H].to(torch.bfloat16)
        tl[: packed.shape[0]] = packed[:, self.n_layers * H :].to(torch.bfloat16)
        return ids.clone(), lm.clone(), th, tl

    def _rollout_batch(self, batch):
        results = []
        for b in range(batch["input_ids"].shape[0]):
            am = batch["attention_mask"][b].bool()
            ids = batch["input_ids"][b][am].clone()
            lm = batch["loss_mask"][b][am].clone()
            results.append(self._rollout_one(ids, lm))
        return results

    @staticmethod
    def _assemble(batch, results):
        T = max(r[0].shape[0] for r in results)
        B = len(results)
        input_ids = torch.zeros(B, T, dtype=batch["input_ids"].dtype)
        attention_mask = torch.zeros(B, T, dtype=batch["attention_mask"].dtype)
        loss_mask = torch.zeros(B, T, dtype=batch["loss_mask"].dtype)
        th = torch.zeros(B, T, results[0][2].shape[-1], dtype=torch.bfloat16)
        tl = torch.zeros(B, T, results[0][3].shape[-1], dtype=torch.bfloat16)
        for b, (seq, lm, th1, tl1) in enumerate(results):
            L = seq.shape[0]
            input_ids[b, :L] = seq
            attention_mask[b, :L] = 1
            loss_mask[b, :L] = lm
            th[b, :L] = th1
            tl[b, :L] = tl1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "opd_th": th,
            "opd_tl": tl,
        }

    def __iter__(self):
        inflight = {}  # future -> cpu_batch(乱序:谁先完成先消费)
        inner = iter(self.dataloader)
        exhausted = False
        zero_wait_streak = 0
        n_yield = 0
        total_wait = 0.0
        t_iter0 = time.time()
        while True:
            while not exhausted and len(inflight) < self.window:
                try:
                    nb = next(inner)
                except StopIteration:
                    exhausted = True
                    break
                inflight[self.executor.submit(self._rollout_batch, nb)] = nb
            if not inflight:
                return
            t0 = time.time()
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            waited = time.time() - t0
            fut = done.pop()
            batch = inflight.pop(fut)
            results = fut.result()  # worker 内已含重试;异常向上抛(可见地崩)
            total_wait += waited
            # ---- 自适应窗口:训练端在等 -> 扩窗;持续零等待 -> 缩窗 ----
            if waited > 0.5:
                zero_wait_streak = 0
                if self.window < self.max_window:
                    self.window += 1
                    print(f"{self._tag} waited {waited:.1f}s -> window={self.window}", flush=True)
                else:
                    # 触顶也要让等待可见,否则阻塞时间在日志里凭空消失
                    print(
                        f"{self._tag} waited {waited:.1f}s (window 已触顶 "
                        f"{self.max_window})",
                        flush=True,
                    )
            else:
                zero_wait_streak += 1
                if zero_wait_streak >= 32 and self.window > self.min_window:
                    self.window -= 1
                    zero_wait_streak = 0
                    print(f"{self._tag} idle streak -> window={self.window}", flush=True)
            n_yield += 1
            if n_yield % 16 == 0:
                el = time.time() - t_iter0
                cs = ""
                if self.cache is not None:
                    h, m, r = self.cache.stats()
                    cs = f" | 缓存 命中{h}/未命中{m} = {r:.0f}%"
                print(
                    f"{self._tag} yielded={n_yield} window={self.window} "
                    f"inflight={len(inflight)} | 累计等待 {total_wait/60:.1f}min / "
                    f"已用 {el/60:.1f}min = {100*total_wait/max(el,1e-9):.0f}%{cs}",
                    flush=True,
                )
            yield self._assemble(batch, results)
