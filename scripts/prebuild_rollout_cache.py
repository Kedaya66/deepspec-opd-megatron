"""阶段一:纯 rollout 预建 gen_ids 缓存(两阶段训练架构的核心)。

架构动机(2026-08-12/13 两天基准实验的最重要教训):
  训练的 EP all_to_all 是同步集合通信,任何 rank 的数据长尾都会让全体等待;
  而 rollout 依赖的特征服务会间歇死锁(魔改 SGLang 已复现 6 次)。二者耦合时,
  服务打嗝 > NCCL 超时窗口 => 整个训练崩(实测三种崩法)。

  解耦:本脚本不起训练、不建 NCCL 组、不占训练 GPU —— 服务死锁只是让它变慢
  (FeatureClient 无限重试 + 断点续跑),绝不会造成任何损失。跑完后训练全程
  warm(0731 实测 5.2x,gbs=512 一步约 40min),训练阶段对服务的依赖降到
  只剩特征导出。

用法(在训练容器内):
  python scripts/prebuild_rollout_cache.py \
      --config config/dspark/dspark_dskv4flash_v5.py \
      --cache-dir /root/rollout_cache_v5 \
      --num-samples 5632 --concurrency 16

  --num-samples 0 表示全数据集。可随时 Ctrl-C,重跑自动跳过已缓存样本。
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=0, help="0=全数据集")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="并发请求数;服务 C=8 即饱和,16 已足够打满")
    ap.add_argument("--start", type=int, default=0, help="起始样本下标(分段跑)")
    ap.add_argument("--perm-seed", type=int, default=42,
                    help="按训练 sampler 的洗牌序(StatelessResumableDistributedSampler,"
                         "seed=42)预建,保证部分预建正好覆盖训练最先消费的样本;"
                         "传 -1 用文件顺序")
    args = ap.parse_args()

    os.environ.setdefault("USE_TORCH", "true")
    import json

    import torch
    from transformers import AutoTokenizer

    from deepspec.data import parser as _parser
    from deepspec.data.feature_client import FeatureClient
    from deepspec.data.parser import preprocess_record
    from deepspec.data.rollout_cache import RolloutCache
    from deepspec.utils import load_config, parse_opts_to_config

    cfg = parse_opts_to_config([], load_config(args.config))
    online = cfg.online
    tokp = str(cfg.model.target_model_name_or_path)
    tok = AutoTokenizer.from_pretrained(tokp, trust_remote_code=True)

    # v4-flash 官方 encode 渲染覆盖(与训练完全同口径)
    enc = os.path.join(tokp, "encoding")
    if enc not in sys.path:
        sys.path.insert(0, enc)
    from encoding_dsv4 import encode_messages

    def _flat(c):
        if isinstance(c, list):
            return "".join(b.get("text", "") for b in c if isinstance(b, dict))
        return c or ""

    _parser.set_render_override(
        lambda t, m, *, add_generation_prompt, enable_thinking=None: encode_messages(
            [dict(x, content=_flat(x.get("content"))) for x in m],
            thinking_mode="chat", add_default_bos_token=True)
    )

    mnt = int(online.opd_max_new_tokens)
    max_len = int(cfg.data.max_length)
    cache = RolloutCache(
        args.cache_dir,
        max_new_tokens=mnt,
        temperature=float(online.opd_temperature),
        top_p=float(online.opd_top_p),
        max_length=max_len,
        weight_version=tokp,
    )
    client = FeatureClient(
        list(online.sglang_server_address),
        timeout=float(online.request_timeout) if "request_timeout" in online else 1200.0,
        tag="prebuild",
    )

    # 收集 prompts(与训练同口径:loss_mask 首个 1 之前为 prompt)
    prompts = []
    path = list(cfg.data.train_data_path)[0]
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            # perm 模式必须全量收集(perm 是对全数据集下标做的);
            # 只有文件序模式才能靠提前 break 省解析时间
            if (args.perm_seed < 0 and args.num_samples
                    and len(prompts) >= args.start + args.num_samples):
                break
            try:
                out = preprocess_record(json.loads(line), tok,
                                        chat_template=str(cfg.data.chat_template),
                                        max_length=max_len)
            except Exception:
                continue
            if out is None:
                continue
            ids = torch.as_tensor(out["input_ids"]).flatten()
            lm = torch.as_tensor(out["loss_mask"]).flatten()
            nz = torch.nonzero(lm, as_tuple=False)
            plen = int(nz[0]) if len(nz) else int(ids.shape[0])
            prompts.append(ids[:plen].tolist())
    # 对齐训练消费顺序:sampler 是全局 randperm(seed)+交错,rank 只是交错切分,
    # 前 K 个"被消费"的样本 = perm[:K]。部分预建必须按这个序,否则训练照样 miss。
    if args.perm_seed >= 0:
        g = torch.Generator()
        g.manual_seed(args.perm_seed)
        perm = torch.randperm(len(prompts), generator=g).tolist()
        prompts = [prompts[i] for i in perm]
    prompts = prompts[args.start:]
    if args.num_samples:
        prompts = prompts[: args.num_samples]
    total = len(prompts)
    print(f"[prebuild] 样本 {total} 条(start={args.start}),并发 {args.concurrency}",
          flush=True)

    done = skip = short = 0
    t0 = time.time()

    def work(prompt):
        budget = max(1, min(mnt, max_len - len(prompt)))
        if cache.get(prompt) is not None:
            return "skip"
        gen_ids, _ = client.rollout(prompt, max_new_tokens=budget,
                                    temperature=float(online.opd_temperature),
                                    top_p=float(online.opd_top_p))
        gen_ids = gen_ids[:budget]
        if len(gen_ids) >= 2:
            cache.put(prompt, gen_ids)
            return "done"
        return "short"  # 训练时走 teacher-forced 回退,无需缓存

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, p) for p in prompts]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()  # FeatureClient 重试耗尽会 raise —— 重跑续传即可
            done += r == "done"
            skip += r == "skip"
            short += r == "short"
            if i % 32 == 0 or i == total:
                el = time.time() - t0
                rate = i / max(el, 1e-9)
                eta = (total - i) / max(rate, 1e-9) / 60
                print(f"[prebuild] {i}/{total} 新建{done} 跳过{skip} 过短{short} "
                      f"| {rate:.2f} 样本/s | 剩余约 {eta:.0f} min", flush=True)

    print(f"[prebuild] 完成: 新建 {done}, 已有 {skip}, 过短 {short}, "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
