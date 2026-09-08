"""并发扫描：不同并发度下 SGLang 取特征的聚合吞吐。
每条 = 一个 /generate 请求（return_hidden_states）。用线程池模拟 C 路并发。
"""
import concurrent.futures as cf
import json
import os
import sys
import time

import requests
import torch

ROOT = "/shenlb/zwf-spec/train-v1/deepspec"
sys.path.insert(0, ROOT)
ENDPOINT = "http://127.0.0.1:30000"
SEQ = int(os.environ.get("SEQ", "1024"))   # 每条截断长度
K = int(os.environ.get("K", "16"))          # 请求条数
LEVELS = [int(x) for x in os.environ.get("LEVELS", "1,2,4,8,16").split(",")]


def _cfg():
    from deepspec.utils import load_config
    return load_config(f"{ROOT}/config/dspark/dspark_dskv4-flash.py")


def load_ids():
    cfg = _cfg()
    mp = str(cfg.model.target_model_name_or_path)
    enc = os.path.join(mp, "encoding")
    sys.path.insert(0, enc)
    from encoding_dsv4 import encode_messages
    from deepspec.data import parser as P

    def flat(c):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict)) if isinstance(c, list) else (c or "")

    P.set_render_override(lambda tok, m, *, add_generation_prompt, enable_thinking=None:
                          encode_messages([dict(x, content=flat(x.get("content"))) for x in m],
                                          thinking_mode="chat", add_default_bos_token=True))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mp)
    from deepspec.data.parser import preprocess_record
    D = "/shenlb/zwf-spec/dataset/dataset-v1/主agent输入_医疗助手_正式环境_perfectblend_openai_conv_regen.jsonl"
    ids = []
    for i, line in enumerate(open(D, encoding="utf-8")):
        if len(ids) >= K:
            break
        out = preprocess_record(json.loads(line), tok, "v4-flash", int(cfg.data.max_length))
        ids.append(out["input_ids"][:SEQ].tolist())
    return ids


def one(ids, sess):
    r = sess.post(f"{ENDPOINT}/generate",
                  json={"input_ids": ids,
                        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                        "return_hidden_states": True},
                  timeout=1200)
    r.raise_for_status()
    hs = r.json()["meta_info"]["hidden_states"]
    import sys as _s; _s.path.insert(0,'/shenlb/zwf-spec/train-v1/deepspec/scripts/data')
    from sglang_target_forward import _decode_hidden
    t = _decode_hidden(hs)           # base64 快路解码
    return t.shape[0]


def main():
    ids_list = load_ids()
    tot_tok = sum(len(x) for x in ids_list)
    print(f"样本 {len(ids_list)} 条, 每条 {SEQ} token, 合计 {tot_tok} token\n")
    # 预热 1 次
    with requests.Session() as s:
        one(ids_list[0], s)
    for C in LEVELS:
        sess = requests.Session()
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=C) as ex:
            list(ex.map(lambda x: one(x, sess), ids_list))
        dt = time.time() - t0
        print(f"并发 C={C:2d}: 墙钟 {dt:6.1f}s | {tot_tok/dt:7.0f} tok/s | {len(ids_list)/dt:5.2f} samples/s")
        sess.close()


if __name__ == "__main__":
    main()
