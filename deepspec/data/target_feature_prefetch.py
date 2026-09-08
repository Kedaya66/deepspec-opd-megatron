"""target 特征的 read-ahead 并发预取(乱序消费 + 自适应窗口 + 多端点轮询)。

移植自 train-v3/deepspec-opd 的 opd_rollout_loader,但去掉 rollout:
payload 换成 teacher-forced 的 fetch_target_features,训练语义与原在线路径完全一致。

为什么需要它 —— 实测(2026-08-10,gbs=512 一步 70.6 min):
    训练卡 74.9% 的采样点 GPU 利用率为 0%,在等特征。
原实现 run_batch 里同步单发:训练卡等 SGLang → SGLang 等训练卡,零重叠。
本 loader 在 read-ahead 窗口内并发把 N 个 batch 的特征打出去,吃 SGLang 的
continuous batching 聚合吞吐,并且**谁先完成谁先消费**,消除队头 straggler 阻塞。

窗口自适应:训练端出现等待就扩窗,持续零等待就缩窗(生产速率略大于消费即可,
窗口开太大只是白白占 CPU 内存)。

⚠️ 两个必须知道的点:
  1. **特征张量留在 CPU**,key 以 `cpu_` 开头。因为单样本特征约
     T×(L+1)×H×2B = 45000×4×7168×2 ≈ 2.6 GB,而实测峰值显存已占卡容量 99.6%
     (只剩 619 MiB)。CUDAPrefetcher 会把 batch 里每个 key 都搬上 GPU 并预取
     一批,`cpu_` 前缀让它跳过 —— 用时在 run_batch 里才 .to(device),
     GPU 占用与改造前完全一致。窗口里的 N 份特征只占主机内存(window=8 时约 20 GB)。
  2. **乱序消费改变了样本→step 的分配**。梯度累积内部求和与顺序无关,所以单步
     数学等价;但跨 step 的样本归属会变,与 StatelessResumableDistributedSampler
     的 next_micro_step 精确续训语义不再逐样本可复现。要求 bit-level 可复现的
     场景把 window 设为 1(退化成串行,等价原行为)。
"""

import itertools
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


class TargetFeaturePrefetchLoader:
    """包在 DataLoader 外:窗口内并发取 target 特征,yield 带特征的 batch。"""

    def __init__(
        self,
        dataloader,
        *,
        endpoints,
        target_layer_ids,
        timeout=1200.0,
        window=8,
        max_window=16,
        min_window=1,
        log_every=64,
    ):
        self.dataloader = dataloader
        self.endpoints = [
            e if str(e).startswith("http") else f"http://{e}"
            for e in (endpoints if isinstance(endpoints, (list, tuple)) else [endpoints])
        ]
        self.target_layer_ids = list(target_layer_ids)
        self.timeout = float(timeout)
        self.window = max(1, int(window))
        self.max_window = max(self.window, int(max_window))
        self.min_window = max(1, int(min_window))
        self.log_every = int(log_every)
        self.executor = ThreadPoolExecutor(max_workers=self.max_window)
        self._rr = itertools.count()  # 端点轮询

    def __len__(self):
        return len(self.dataloader)

    def _fetch_one(self, batch):
        """线程池内跑(含 HTTP)。返回 CPU 上的 (th, tl)。"""
        from deepspec.trainer.dspark_online_trainer import fetch_target_features

        endpoint = self.endpoints[next(self._rr) % len(self.endpoints)]
        return fetch_target_features(
            endpoint=endpoint,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_layer_ids=self.target_layer_ids,
            timeout=self.timeout,
        )

    def __iter__(self):
        inflight = {}          # future -> cpu_batch
        inner = iter(self.dataloader)
        exhausted = False
        zero_wait_streak = 0
        n_yield = 0
        while True:
            # 补满窗口
            while not exhausted and len(inflight) < self.window:
                try:
                    nb = next(inner)
                except StopIteration:
                    exhausted = True
                    break
                inflight[self.executor.submit(self._fetch_one, nb)] = nb
            if not inflight:
                return

            t0 = time.time()
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            waited = time.time() - t0
            fut = done.pop()
            batch = inflight.pop(fut)
            th, tl = fut.result()   # fetch_target_features 内已含重试;异常向上抛

            # 自适应窗口:训练端在等 -> 扩;持续零等待 -> 缩
            if waited > 0.5:
                zero_wait_streak = 0
                if self.window < self.max_window:
                    self.window += 1
                    print(
                        f"[feat-prefetch] waited {waited:.1f}s -> window={self.window}",
                        flush=True,
                    )
            else:
                zero_wait_streak += 1
                if zero_wait_streak >= 32 and self.window > self.min_window:
                    self.window -= 1
                    zero_wait_streak = 0
                    print(
                        f"[feat-prefetch] idle streak -> window={self.window}", flush=True
                    )

            n_yield += 1
            if self.log_every and n_yield % self.log_every == 0:
                print(
                    f"[feat-prefetch] yielded={n_yield} window={self.window} "
                    f"inflight={len(inflight)} endpoints={len(self.endpoints)}",
                    flush=True,
                )

            out = dict(batch)
            # cpu_ 前缀:CUDAPrefetcher 不搬这两个,避免多占一份 ~2.6GB 显存
            out["cpu_target_hidden_states"] = th
            out["cpu_target_last_hidden_states"] = tl
            yield out
