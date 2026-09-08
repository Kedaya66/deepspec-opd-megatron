"""进程内 Engine 取特征计时（对比 HTTP）。真实完整样本，无 JSON。"""
import os
import sys
import time

import torch

ROOT = "/shenlb/zwf-spec/train-v1/deepspec"
sys.path.insert(0, ROOT)
LAYERS = [40, 41, 42]
H = 4096


def main():
    os.environ["SGLANG_DSPARK_EXPORT_LAYERS"] = ",".join(map(str, LAYERS))
    from deepspec.utils import load_config
    cfg = load_config(f"{ROOT}/config/dspark/dspark_dskv4-flash.py")
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
    import json
    D = "/shenlb/zwf-spec/dataset/dataset-v1/主agent输入_医疗助手_正式环境_perfectblend_openai_conv_regen.jsonl"
    samples = []
    for i, line in enumerate(open(D, encoding="utf-8")):
        if len(samples) >= 3:
            break
        ids = preprocess_record(json.loads(line), tok, "v4-flash", int(cfg.data.max_length))["input_ids"].tolist()
        samples.append(ids)

    import sglang as sgl
    t_load = time.time()
    eng = sgl.Engine(
        model_path=mp, tp_size=8, trust_remote_code=True, mem_fraction_static=0.85,
        skip_tokenizer_init=True, disable_cuda_graph=True, enable_return_hidden_states=True,
        moe_runner_backend="marlin", disable_radix_cache=True, chunked_prefill_size=102400,
    )
    print(f"[engine] 加载+warmup {time.time()-t_load:.1f}s")

    # 预热一次（不计时）
    eng.generate(input_ids=samples[0][:64], sampling_params={"max_new_tokens": 1, "temperature": 0.0}, return_hidden_states=True)

    for k, ids in enumerate(samples):
        L = len(ids)
        t0 = time.time()
        r = eng.generate(input_ids=ids, sampling_params={"max_new_tokens": 1, "temperature": 0.0}, return_hidden_states=True)
        hs = r["meta_info"]["hidden_states"]
        t_gen = time.time() - t0
        # 转张量（客户端消费开销也计入，和 HTTP 口径一致）
        t1 = time.time()
        packed = hs if isinstance(hs, torch.Tensor) else torch.as_tensor(hs)
        if packed.dim() == 3:
            packed = packed.squeeze(0)
        t_cast = time.time() - t1
        ret = packed.shape[0]
        print(f"样本{k}: L={L} 返回={ret} full={ret==L} | generate={t_gen:.1f}s cast={t_cast:.1f}s 合计={t_gen+t_cast:.1f}s -> {L/(t_gen+t_cast):.0f} tok/s")
    eng.shutdown()


if __name__ == "__main__":
    main()
