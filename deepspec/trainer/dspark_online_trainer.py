"""在线 DSpark 训练:每个 batch 现调 SGLang 服务取 target 特征,不读离线 cache。

与离线 dspark_trainer.py 的区别只在数据侧:
  * 数据源:原始对话 JSONL(JsonLineDataset)+ ConversationCollator
    -> 每 batch 产出 input_ids / attention_mask / loss_mask(无 target 特征)。
  * run_batch:调 SGLang(带 SGLANG_DSPARK_EXPORT_LAYERS 的服务)现取
    target_hidden_states / target_last_hidden_states,内存消费即弃。

需要的 config 字段:
    data.train_data_path        : str | list[str]   # 对话 JSONL 路径
    data.min_loss_tokens        : int (默认 0)
    online.sglang_server_address: str | list[str]   # 如 "127.0.0.1:30000"
    online.request_timeout      : float (可选)
target_layer_ids / chat_template / max_length 复用 model/data 段既有字段。
"""
import base64
import time
import os
import sys

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader

from deepspec.data import ConversationCollator
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.utils import (
    StatelessResumableDistributedSampler,
    print_on_global_main,
)
from deepspec.trainer.dspark_trainer import DeepSeekV4FlashDSparkTrainer


# --------------------------------------------------------------------------
# 内联的 SGLang 取特征客户端(base64 快路解码,与 scripts/data 版一致)
# --------------------------------------------------------------------------
# _decode_hidden 的实现已下沉到 deepspec/data/feature_client.py(2026-08-13 重构),
# 这里保留别名以兼容旧引用(probe 脚本、外部工具)。
from deepspec.data.feature_client import decode_hidden as _decode_hidden  # noqa: E402


def fetch_target_features(
    *, endpoint, input_ids, attention_mask, target_layer_ids, timeout=1200.0,
    max_retries=120,
):
    """逐样本取 teacher-forced 特征,拼回 [B,T,·] bf16。(重构:走 FeatureClient)"""
    from deepspec.data.feature_client import FeatureClient

    client = FeatureClient(endpoint, timeout=timeout, max_retries=max_retries,
                           tag="dspark-online")
    n_layers = len([int(x) for x in target_layer_ids])
    B, T = input_ids.shape
    th_list, tl_list = [], []
    for b in range(B):
        mask_b = attention_mask[b].bool()
        ids = input_ids[b][mask_b].tolist()
        L = len(ids)
        packed = client.prefill_features(ids, rows=L)
        H = packed.shape[-1] // (n_layers + 1)
        th = torch.zeros(T, n_layers * H, dtype=torch.bfloat16)
        tl = torch.zeros(T, H, dtype=torch.bfloat16)
        th[:L] = packed[:, : n_layers * H].to(torch.bfloat16)
        tl[:L] = packed[:, n_layers * H :].to(torch.bfloat16)
        th_list.append(th)
        tl_list.append(tl)
    return torch.stack(th_list, 0), torch.stack(tl_list, 0)


def rollout_target_features(
    *, endpoint, prompt_ids, max_new_tokens, temperature=0.0, top_p=1.0,
    timeout=1200.0, max_retries=120,
):
    """OPD:target 对 prompt 自回归 rollout,一趟拿到 生成ids + 全程特征。

    返回 (gen_ids, packed[L_prompt+gen-1, (n+1)*H])。(重构:走 FeatureClient)
    """
    from deepspec.data.feature_client import FeatureClient

    client = FeatureClient(endpoint, timeout=timeout, max_retries=max_retries,
                           tag="dspark-opd")
    return client.rollout(
        prompt_ids, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
    )


def _v4_render_worker_init(worker_id):
    """DataLoader worker 初始化:重装 V4-Flash 官方 encode 渲染覆盖。

    train.py 用 spawn 启动,worker 会重新 import parser(渲染覆盖丢失),
    故每个 worker 启动时按 DSPARK_TARGET_MODEL_PATH 重新安装。
    """
    import os as _os
    import sys as _sys

    mp = _os.environ.get("DSPARK_TARGET_MODEL_PATH", "")
    if not mp:
        return
    enc = _os.path.join(mp, "encoding")
    if enc not in _sys.path:
        _sys.path.insert(0, enc)
    from encoding_dsv4 import encode_messages
    from deepspec.data import parser as _parser

    def _flat(c):
        if isinstance(c, list):
            return "".join(b.get("text", "") for b in c if isinstance(b, dict))
        return c or ""

    def _render(tok, messages, *, add_generation_prompt, enable_thinking=None):
        msgs = [dict(m, content=_flat(m.get("content"))) for m in messages]
        return encode_messages(msgs, thinking_mode="chat", add_default_bos_token=True)

    _parser.set_render_override(_render)


def _cfg(obj, key, default=None):
    return obj[key] if (obj is not None and key in obj) else default


# --------------------------------------------------------------------------
# 在线训练 mixin:替换数据源 + run_batch 现取特征
# --------------------------------------------------------------------------
class DSparkOnlineTrainerMixin:
    # 占位;在线不走离线 CacheCollator,dataloader 自己构造 collator
    data_collator_cls = ConversationCollator
    _data_worker_init = None  # 子类可设为 spawn-safe 的 worker_init_fn

    def _setup_data_encoding(self):
        """子类可覆写:安装 chat_template 渲染覆盖(如 v4-flash 官方 encode)。"""

    def _build_train_dataset(self):
        self._setup_data_encoding()
        paths = self.args.data.train_data_path
        if isinstance(paths, str):
            paths = [paths]
        return JsonLineDataset(data_paths=list(paths))

    def _validate_train_dataset(self):
        # 在线无离线 cache 可校验
        print_on_global_main("[dspark-online] token dataset ready; skip cache validation.")

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            total_size=self.samples_per_epoch,
            start_global_offset_samples=start_offset_samples,
            num_samples=num_samples,
        )
        collator = ConversationCollator(
            tokenizer=self.tokenizer,
            chat_template=self.args.data.chat_template,
            max_length=self.args.data.max_length,
            # 默认 0:保证每个 batch 非空(否则 CUDAPrefetcher 遇 None 会崩)
            min_loss_tokens=int(_cfg(self.args.data, "min_loss_tokens", 0)),
        )
        loader = DataLoader(
            self.train_dataset,
            batch_size=int(self.args.train.local_batch_size),
            sampler=sampler,
            collate_fn=collator,
            num_workers=int(self.args.data.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
            worker_init_fn=self._data_worker_init,
        )
        online = self.args.online if "online" in self.args else None
        if bool(_cfg(online, "opd_rollout", False)):
            # OPD:并发 rollout 预取(read-ahead 窗口内并发打特征服务,
            # 吃 SGLang continuous batching 的聚合吞吐),yield 纯张量 batch。
            from deepspec.data.opd_rollout_loader import RolloutPrefetchLoader

            addrs = _cfg(online, "sglang_server_address", "127.0.0.1:30000")
            if isinstance(addrs, str):
                addrs = [addrs]
            eps = [a if str(a).startswith("http") else f"http://{a}" for a in addrs]
            loader = RolloutPrefetchLoader(
                loader,
                endpoints=eps,
                n_target_layers=len(list(self.args.model.target_layer_ids)),
                max_length=int(self.args.data.max_length),
                max_new_tokens=int(_cfg(online, "opd_max_new_tokens", 512)),
                temperature=float(_cfg(online, "opd_temperature", 0.0)),
                top_p=float(_cfg(online, "opd_top_p", 1.0)),
                timeout=float(_cfg(online, "request_timeout", 1200.0)),
                prefetch=int(_cfg(online, "opd_prefetch", 8)),
                max_prefetch=int(_cfg(online, "opd_max_prefetch", 16)),
                cache_dir=_cfg(online, "opd_rollout_cache_dir", None),
                # target 模型路径进缓存 key:换权重目录 => 缓存自动失效
                weight_version=str(self.args.model.target_model_name_or_path),
            )
        elif int(_cfg(online, "feature_prefetch", 0)) > 0:
            # teacher-forced 的并发预取(与 OPD 分支互斥:两者占同一个位置)。
            # 实测 2026-08-10 于 train-v1/deepspec-megatron:gbs=512 一步
            # 70.6 → 37.5 min,loss 3.9446 → 3.9445(乱序消费不改梯度累积的和)。
            from deepspec.data.target_feature_prefetch import (
                TargetFeaturePrefetchLoader,
            )

            window = int(_cfg(online, "feature_prefetch", 8))
            eps = self._sglang_endpoints()
            print_on_global_main(
                f"[dspark-online] feature prefetch on: window={window} endpoints={eps}"
            )
            loader = TargetFeaturePrefetchLoader(
                loader,
                endpoints=eps,
                target_layer_ids=list(self.args.model.target_layer_ids),
                timeout=float(_cfg(online, "request_timeout", 1200.0)),
                window=window,
                max_window=int(_cfg(online, "feature_prefetch_max", max(window, 16))),
            )
        return loader

    def _sglang_endpoints(self):
        """全部特征服务地址(预取按样本轮询分流)。原 _sglang_endpoint 只取 [0],
        配了多实例也用不上。"""
        online = self.args.online if "online" in self.args else None
        addrs = _cfg(online, "sglang_server_address", "127.0.0.1:30000")
        if isinstance(addrs, str):
            addrs = [addrs]
        return [a if str(a).startswith("http") else f"http://{a}" for a in addrs]

    def _sglang_endpoint(self):
        return self._sglang_endpoints()[0]

    def _opd_rebuild_batch(self, batch, online):
        """OPD:每样本取 prompt(首个 loss token 之前),让 target rollout 自产
        轨迹,重建 input_ids/loss_mask = prompt+轨迹,特征来自同一趟前向。
        注意 packed 行数 = 序列长-1(最后生成 token 无 hidden),loss 区据此截止。"""
        n_layers = len(list(self.args.model.target_layer_ids))
        dev = batch["input_ids"].device
        B, T = batch["input_ids"].shape
        max_len = int(self.args.data.max_length)
        new_ids = torch.zeros(B, T, dtype=batch["input_ids"].dtype)
        new_am = torch.zeros(B, T, dtype=batch["attention_mask"].dtype)
        new_lm = torch.zeros(B, T, dtype=batch["loss_mask"].dtype)
        th = tl = None
        for b in range(B):
            am = batch["attention_mask"][b].bool()
            ids = batch["input_ids"][b][am]
            lm = batch["loss_mask"][b][am]
            nz = torch.nonzero(lm, as_tuple=False)
            prompt_len = int(nz[0]) if len(nz) else int(ids.shape[0])
            prompt = ids[:prompt_len].tolist()
            budget = max(
                0,
                min(
                    int(_cfg(online, "opd_max_new_tokens", 1024)),
                    max_len - prompt_len,
                    T - prompt_len,
                ),
            )
            gen_ids, packed = rollout_target_features(
                endpoint=self._sglang_endpoint(),
                prompt_ids=prompt,
                max_new_tokens=max(budget, 1),
                temperature=float(_cfg(online, "opd_temperature", 0.0)),
                top_p=float(_cfg(online, "opd_top_p", 1.0)),
                timeout=float(_cfg(online, "request_timeout", 1200.0)),
            )
            gen_ids = gen_ids[:budget]
            H = packed.shape[-1] // (n_layers + 1)
            if th is None:
                th = torch.zeros(B, T, n_layers * H, dtype=torch.bfloat16)
                tl = torch.zeros(B, T, H, dtype=torch.bfloat16)
            rows = min(int(packed.shape[0]), T)
            th[b, :rows] = packed[:rows, : n_layers * H].to(torch.bfloat16)
            tl[b, :rows] = packed[:rows, n_layers * H :].to(torch.bfloat16)
            seq = (prompt + gen_ids)[:T]
            new_ids[b, : len(seq)] = torch.as_tensor(seq, dtype=new_ids.dtype)
            new_am[b, : len(seq)] = 1
            hi = min(len(seq), rows)  # 无 hidden 的末位置不进 loss 区
            if hi > prompt_len:
                new_lm[b, prompt_len:hi] = 1
        batch = dict(batch)
        batch["input_ids"] = new_ids.to(dev)
        batch["attention_mask"] = new_am.to(dev)
        batch["loss_mask"] = new_lm.to(dev)
        return batch, th, tl

    def run_batch(self, batch):
        online = self.args.online if "online" in self.args else None
        if "opd_th" in batch:
            # 并发预取 loader 已重建 batch 并带好特征(GPU 上)
            th = batch["opd_th"]
            tl = batch["opd_tl"]
        elif "cpu_target_hidden_states" in batch:
            # teacher-forced 预取:特征留在主机内存(单样本约 2.6 GB,峰值显存
            # 已占卡 99.6%,跟着 CUDAPrefetcher 搬会多占一份必 OOM),此处才上卡
            th = batch["cpu_target_hidden_states"]
            tl = batch["cpu_target_last_hidden_states"]
        elif bool(_cfg(online, "opd_rollout", False)):
            batch, th, tl = self._opd_rebuild_batch(batch, online)
        else:
            th, tl = fetch_target_features(
                endpoint=self._sglang_endpoint(),
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target_layer_ids=list(self.args.model.target_layer_ids),
                timeout=float(_cfg(online, "request_timeout", 1200.0)),
            )
        th = th.to(self.device, dtype=self.precision_dtype)
        tl = tl.to(self.device, dtype=self.precision_dtype)
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=th,
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=tl,
        )
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )
        return loss


class DeepSeekV4FlashDSparkOnlineTrainer(
    DSparkOnlineTrainerMixin, DeepSeekV4FlashDSparkTrainer
):
    _data_worker_init = staticmethod(_v4_render_worker_init)
    """DeepSeek-V4-Flash 在线 DSpark 训练器。

    draft = DeepSeekV4FlashDSparkModel(+可选预训练权重),
    target 特征由 SGLang 服务(SGLANG_DSPARK_EXPORT_LAYERS=40,41,42)在线提供。
    """

    def _setup_data_encoding(self):
        # V4-Flash 无 jinja chat_template:安装官方 encode_messages(chat 模式)渲染覆盖
        mp = str(self.args.model.target_model_name_or_path)
        enc = os.path.join(mp, "encoding")
        if enc not in sys.path:
            sys.path.insert(0, enc)
        from encoding_dsv4 import encode_messages
        from deepspec.data import parser as _parser

        def _flat(c):
            if isinstance(c, list):
                return "".join(b.get("text", "") for b in c if isinstance(b, dict))
            return c or ""

        def _render(tok, messages, *, add_generation_prompt, enable_thinking=None):
            msgs = [dict(m, content=_flat(m.get("content"))) for m in messages]
            return encode_messages(
                msgs, thinking_mode="chat", add_default_bos_token=True
            )

        os.environ["DSPARK_TARGET_MODEL_PATH"] = mp
        _parser.set_render_override(_render)
        print_on_global_main("[dspark-online] installed V4-Flash official encode_messages render.")


__all__ = [
    "DSparkOnlineTrainerMixin",
    "DeepSeekV4FlashDSparkOnlineTrainer",
    "fetch_target_features",
]
