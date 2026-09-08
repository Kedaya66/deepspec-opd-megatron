"""校验：SGLang dspark-export 特征 与 HF hook 特征 是否等价。

分 4 个独立进程运行（避免 HF 与 SGLang 同时占卡）：
  python _validate_sglang_vs_hf.py prep      # 分词 K 条样本，存 input_ids
  python _validate_sglang_vs_hf.py sglang    # 起 sgl.Engine + export，取特征
  python _validate_sglang_vs_hf.py hf        # HF device_map + hook，取参考特征
  python _validate_sglang_vs_hf.py compare   # 逐层 cos/mae/rel_l2 比对
"""
import argparse
import json
import os
import sys

import torch

ROOT = "/shenlb/zwf-spec/train-v1/deepspec"
sys.path.insert(0, ROOT)

DATASET = "/shenlb/zwf-spec/dataset/dataset-v1/主agent输入_医疗助手_正式环境_perfectblend_openai_conv_regen.jsonl"
VAL_DIR = os.path.expanduser("~/.cache/deepspec/_val")
IDS_PATH = os.path.join(VAL_DIR, "input_ids.json")
SGL_PATH = os.path.join(VAL_DIR, "sglang.pt")
HF_PATH = os.path.join(VAL_DIR, "hf.pt")

K = 3            # 样本数
MAXTOK = 512     # 每条截断长度（比对够用，省显存/时间）
LAYERS = [40, 41, 42]
H = 4096
NAMES = ["L40", "L41", "L42", "last"]


def load_cfg():
    from deepspec.utils import load_config
    return load_config(f"{ROOT}/config/dspark/dspark_dskv4-flash.py")


def _install_official_render(mp):
    """复刻 debug 脚本里的官方 encode_messages 覆盖。"""
    enc = os.path.join(mp, "encoding")
    if enc not in sys.path:
        sys.path.insert(0, enc)
    from encoding_dsv4 import encode_messages
    from deepspec.data import parser as P

    def _flat(c):
        if isinstance(c, list):
            return "".join(b.get("text", "") for b in c if isinstance(b, dict))
        return c or ""

    def _render(tok, messages, *, add_generation_prompt, enable_thinking=None):
        msgs = [dict(m, content=_flat(m.get("content"))) for m in messages]
        return encode_messages(msgs, thinking_mode="chat", add_default_bos_token=True)

    P.set_render_override(_render)


def prep():
    cfg = load_cfg()
    mp = str(cfg.model.target_model_name_or_path)
    from transformers import AutoTokenizer
    _install_official_render(mp)
    tok = AutoTokenizer.from_pretrained(mp)
    from deepspec.data.parser import preprocess_record

    ids_list = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            if len(ids_list) >= K:
                break
            rec = json.loads(line)
            out = preprocess_record(rec, tok, "v4-flash", int(cfg.data.max_length))
            ids = out["input_ids"][:MAXTOK].tolist()
            ids_list.append(ids)
    os.makedirs(VAL_DIR, exist_ok=True)
    with open(IDS_PATH, "w") as f:
        json.dump(ids_list, f)
    print(f"[prep] saved {len(ids_list)} 条, lens={[len(x) for x in ids_list]}")


def run_sglang():
    os.environ["SGLANG_DSPARK_EXPORT_LAYERS"] = ",".join(map(str, LAYERS))
    cfg = load_cfg()
    mp = str(cfg.model.target_model_name_or_path)
    import sglang as sgl

    ids_list = json.load(open(IDS_PATH))
    engine = sgl.Engine(
        model_path=mp,
        tp_size=8,
        trust_remote_code=True,
        mem_fraction_static=0.85,
        skip_tokenizer_init=True,
        disable_cuda_graph=True,
        enable_return_hidden_states=True,
        moe_runner_backend="marlin",  # V4-Flash MoE 需 Marlin runner（否则 fused_moe Hidden size mismatch）—— 照搬团队 run_sglang_dspark.sh
        disable_radix_cache=True,  # 取特征必须关前缀缓存：否则共享前缀命中缓存，只返回后缀 token 的 hidden states  # 只做 prefill 取特征，无需 decode CUDA graph；且 export 改了 hidden 维度会触发 capture 断言
    )
    outs = []
    for k, ids in enumerate(ids_list):
        r = engine.generate(
            input_ids=ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            return_hidden_states=True,
        )
        hs = r["meta_info"]["hidden_states"]
        print(f"[sglang] sample{k} hidden_states type={type(hs).__name__}")
        t = hs if isinstance(hs, torch.Tensor) else torch.as_tensor(hs)
        t = t.float()
        print(f"[sglang] sample{k} raw shape={tuple(t.shape)} (期望 [~{len(ids)}, {(len(LAYERS)+1)*H}])")
        t = t[: len(ids)]
        outs.append(t.cpu())
    engine.shutdown()
    torch.save(outs, SGL_PATH)
    print("[sglang] saved ->", SGL_PATH)


def _hook(store, layer_id):
    def fn(_m, _i, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        store[layer_id] = t.detach()
    return fn


def run_hf():
    cfg = load_cfg()
    mp = str(cfg.model.target_model_name_or_path)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        mp, dtype=torch.bfloat16, attn_implementation="eager", device_map="auto"
    ).eval()
    backbone = getattr(model, "model", model)
    layers = backbone.layers
    dev0 = backbone.embed_tokens.weight.device

    ids_list = json.load(open(IDS_PATH))
    outs = []
    for k, ids in enumerate(ids_list):
        input_ids = torch.tensor([ids], device=dev0)
        attn = torch.ones_like(input_ids)
        cap = {}
        handles = [layers[l].register_forward_hook(_hook(cap, l)) for l in LAYERS]
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        last = out.last_hidden_state.detach()
        th = torch.cat([cap[l].to(last.device) for l in LAYERS], dim=-1)  # [1,L,3H]
        packed = torch.cat([th, last], dim=-1)[0]  # [L, 4H]
        for hd in handles:
            hd.remove()
        print(f"[hf] sample{k} packed shape={tuple(packed.shape)}")
        outs.append(packed.float().cpu())
    torch.save(outs, HF_PATH)
    print("[hf] saved ->", HF_PATH)


def compare():
    sg = torch.load(SGL_PATH)
    hf = torch.load(HF_PATH)
    print(f"比对 {len(sg)} 条样本；每条按 {NAMES} 四块（各 {H} 维）\n")
    all_cos = []
    for i, (a, b) in enumerate(zip(sg, hf)):
        L = min(a.shape[0], b.shape[0])
        a, b = a[:L], b[:L]
        print(f"#样本 {i}: sglang {tuple(a.shape)} vs hf {tuple(b.shape)} (取前 {L} token)")
        for k, name in enumerate(NAMES):
            av = a[:, k * H:(k + 1) * H]
            bv = b[:, k * H:(k + 1) * H]
            cos = torch.nn.functional.cosine_similarity(av, bv, dim=-1).mean().item()
            mae = (av - bv).abs().mean().item()
            rel = ((av - bv).norm() / (bv.norm() + 1e-6)).item()
            all_cos.append(cos)
            flag = "✅" if cos > 0.99 else ("⚠️" if cos > 0.95 else "❌")
            print(f"   {name}: cos={cos:.4f} mae={mae:.4f} rel_l2={rel:.4f} {flag}")
        print()
    m = sum(all_cos) / len(all_cos)
    print(f"总体平均 cosine = {m:.4f} ->", "✅ 可视为等价" if m > 0.99 else ("⚠️ 接近但有偏差" if m > 0.95 else "❌ 不等价，需排查"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "sglang", "hf", "compare"])
    fn = {"prep": prep, "sglang": run_sglang, "hf": run_hf, "compare": compare}
    fn[ap.parse_args().mode]()
