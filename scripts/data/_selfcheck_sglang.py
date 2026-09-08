"""SGLang export 特征 轻量自洽验证（不依赖 HF）：
  1) 数值健康：shape / 是否全长 / NaN / Inf / 各块范数
  2) tp8 vs tp2 一致性：逐块 cos/mae（顺带坐实 2 卡可部署）

用法：
  python _selfcheck_sglang.py run --tp 8 --out ~/.cache/deepspec/_val/sglang_tp8.pt          # 8 卡
  CUDA_VISIBLE_DEVICES=0,1 python _selfcheck_sglang.py run --tp 2 --out .../sglang_tp2.pt      # 2 卡
  python _selfcheck_sglang.py selfcompare
"""
import argparse
import json
import os
import sys

import torch

ROOT = "/shenlb/zwf-spec/train-v1/deepspec"
sys.path.insert(0, ROOT)
VAL = os.path.expanduser("~/.cache/deepspec/_val")
IDS = os.path.join(VAL, "input_ids.json")
LAYERS = [40, 41, 42]
H = 4096
NAMES = ["L40", "L41", "L42", "last"]


def _cfg():
    from deepspec.utils import load_config
    return load_config(f"{ROOT}/config/dspark/dspark_dskv4-flash.py")


def run(tp, out):
    os.environ["SGLANG_DSPARK_EXPORT_LAYERS"] = ",".join(map(str, LAYERS))
    mp = str(_cfg().model.target_model_name_or_path)
    import sglang as sgl

    ids_list = json.load(open(IDS))
    eng = sgl.Engine(
        model_path=mp,
        tp_size=tp,
        trust_remote_code=True,
        mem_fraction_static=0.85,
        skip_tokenizer_init=True,
        disable_cuda_graph=True,
        enable_return_hidden_states=True,
        moe_runner_backend="marlin",
        disable_radix_cache=True,
    )
    outs = []
    for k, ids in enumerate(ids_list):
        r = eng.generate(
            input_ids=ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            return_hidden_states=True,
        )
        hs = r["meta_info"]["hidden_states"]
        t = (hs if isinstance(hs, torch.Tensor) else torch.as_tensor(hs)).float()
        if t.dim() == 3:
            t = t.squeeze(0)
        t = t[: len(ids)]
        nan = bool(torch.isnan(t).any())
        inf = bool(torch.isinf(t).any())
        norms = [t[:, i * H:(i + 1) * H].norm(dim=-1).mean().item() for i in range(4)]
        full = t.shape[0] == len(ids)
        print(
            f"[tp{tp}] s{k} shape={tuple(t.shape)} full={full} nan={nan} inf={inf} "
            f"blocknorm={['%.1f' % x for x in norms]}"
        )
        outs.append(t.cpu())
    eng.shutdown()
    torch.save(outs, out)
    print(f"[tp{tp}] saved -> {out}")


def selfcompare(a, b):
    A = torch.load(a)
    B = torch.load(b)
    print(f"比对 A={os.path.basename(a)}  vs  B={os.path.basename(b)}\n")
    cos_all = []
    for i, (x, y) in enumerate(zip(A, B)):
        L = min(x.shape[0], y.shape[0])
        x, y = x[:L], y[:L]
        print(f"#样本 {i}: {tuple(x.shape)} vs {tuple(y.shape)} (取前 {L})")
        for k, nm in enumerate(NAMES):
            xv = x[:, k * H:(k + 1) * H]
            yv = y[:, k * H:(k + 1) * H]
            c = torch.nn.functional.cosine_similarity(xv, yv, dim=-1).mean().item()
            m = (xv - yv).abs().mean().item()
            cos_all.append(c)
            flag = "✅" if c > 0.999 else ("⚠️" if c > 0.99 else "❌")
            print(f"   {nm}: cos={c:.5f} mae={m:.4f} {flag}")
        print()
    mm = sum(cos_all) / len(cos_all)
    print(f"平均 cos={mm:.5f} ->", "✅ tp8/tp2 一致" if mm > 0.999 else ("⚠️ 近似有偏差" if mm > 0.99 else "❌ 不一致"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "selfcompare"])
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(VAL, "sglang_tp8.pt"))
    ap.add_argument("--a", default=os.path.join(VAL, "sglang_tp8.pt"))
    ap.add_argument("--b", default=os.path.join(VAL, "sglang_tp2.pt"))
    args = ap.parse_args()
    if args.mode == "run":
        run(args.tp, args.out)
    else:
        selfcompare(args.a, args.b)
