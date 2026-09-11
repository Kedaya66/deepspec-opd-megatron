"""训练 ckpt(mcore serving-style 布局)-> serving draft(mtp.* 布局),修正版。

相对旧版修复两处 critical(2026-09-10 实测坐实):
  (1) rope 通道逆置换:训练加载时对 wq_b/wkv/kv_norm/wo_a 施加了 interleaved->rotate_half
      置换(load_pretrained_match_v1.ROPE_PERM),导出时必须用 inv_perm=argsort(ROPE_PERM)
      逆回,否则 serving 的 rope kernel(interleaved)读到错排通道 -> 注意力全崩。
  (2) 头部键归属:final norm / hc_head / markov_head / confidence_head 属于【末 stage】
      (mtp.{L-1}),旧版错放到 mtp.0 -> serving 只在末 stage 建这些模块,导致 norm=默认全1、
      hc_head=未初始化内存。hc_head 叶名用【下划线】hc_head_{base,fn,scale}(匹配预训练)。
  另:oracle 从"只查形状"升级为"查形状+查覆盖率"(旧版两 bug 双双漏过)。

reshape/置换配方均经 verify_conversion.py 实测:对预训练同键施加同变换后 rel diff ~8e-4(bf16 级)。
"""
import argparse, glob, json, os, re, shutil
import torch
from safetensors.torch import load_file, save_file

PRETRAIN = "/shenlb/zwf-spec/train/model/DeepSeek-V4-Flash-DSpark-bf16"
ROPE_PERM = torch.cat([torch.arange(0, 64, 2), torch.arange(1, 64, 2)])
INV_PERM = torch.argsort(ROPE_PERM)   # 逆置换:rotate_half -> interleaved
NOPE = 512 - 64  # 448;rope 通道 = head_dim 末 64 维

def _unperm_wq_b(t):        # (32768,1024) view(64,512,-1),逆置换 dim1 末64
    w = t.view(64, 512, -1).clone(); w[:, NOPE:] = w[:, NOPE:][:, INV_PERM]; return w.reshape(t.shape)
def _unperm_row512(t):      # (512,*) 或 (512,) 末64 行逆置换
    t = t.clone(); t[NOPE:] = t[NOPE:][INV_PERM]; return t
def _unperm_wo_a_2d(t):     # serving 布局 (8192,4096):rope 通道在末维(4096=8*512),逆置换每512块末64
    g = t.shape[1] // 512   # 8
    w = t.view(t.shape[0], g, 512).clone(); w[..., NOPE:] = w[..., NOPE:][..., INV_PERM]; return w.reshape(t.shape)

def to_serving(mc):
    out = {}
    last = max(int(m.group(1)) for k in mc if (m := re.match(r"layers\.(\d+)\.", k)))
    pat_e = re.compile(r"^layers\.(\d+)\.ffn\.experts\.linear_fc([12])\.weight(\d+)$")
    for k, v in mc.items():
        if "_extra_state" in k:
            continue
        m = pat_e.match(k)
        if m:
            li, fc, ge = m.groups()
            if fc == "1":
                g, u = v.chunk(2, dim=0)
                out[f"mtp.{li}.ffn.experts.{ge}.w1.weight"] = g.contiguous()
                out[f"mtp.{li}.ffn.experts.{ge}.w3.weight"] = u.contiguous()
            else:
                out[f"mtp.{li}.ffn.experts.{ge}.w2.weight"] = v.clone()
            continue
        lm = re.match(r"^layers\.(\d+)\.(.+)$", k)
        if lm:
            li, rest = lm.groups()
            if rest == "ffn.router.weight":
                out[f"mtp.{li}.ffn.gate.weight"] = v.clone()
            elif rest == "ffn.router.expert_bias":
                out[f"mtp.{li}.ffn.gate.bias"] = v.clone()
            elif rest == "ffn.shared_experts.linear_fc1.weight":
                g, u = v.chunk(2, dim=0)
                out[f"mtp.{li}.ffn.shared_experts.w1.weight"] = g.contiguous()
                out[f"mtp.{li}.ffn.shared_experts.w3.weight"] = u.contiguous()
            elif rest == "ffn.shared_experts.linear_fc2.weight":
                out[f"mtp.{li}.ffn.shared_experts.w2.weight"] = v.clone()
            elif rest == "attn.wo_a":
                w2d = v.reshape(v.shape[0] * v.shape[1], -1).contiguous()
                out[f"mtp.{li}.attn.wo_a.weight"] = _unperm_wo_a_2d(w2d)   # rope 逆置换
            elif rest == "attn.wq_b.weight":
                out[f"mtp.{li}.attn.wq_b.weight"] = _unperm_wq_b(v)        # rope 逆置换
            elif rest == "attn.wkv.weight":
                out[f"mtp.{li}.attn.wkv.weight"] = _unperm_row512(v)       # rope 逆置换
            elif rest == "attn.kv_norm.weight":
                out[f"mtp.{li}.attn.kv_norm.weight"] = _unperm_row512(v)   # rope 逆置换
            else:
                out[f"mtp.{li}.{rest}"] = v.clone()
            continue
        # 顶层单例
        if k in ("embed.weight", "head.weight"):
            out[k] = v.clone()
        elif k == "fc.weight":
            out["mtp.0.main_proj.weight"] = v.clone()
        elif k == "hidden_norm.weight":
            out["mtp.0.main_norm.weight"] = v.clone()
        elif k == "norm.weight":
            out[f"mtp.{last}.norm.weight"] = v.clone()                     # 末 stage
        elif k.startswith("hc_head."):
            leaf = k.split(".", 1)[1]                                       # base/fn/scale
            out[f"mtp.{last}.hc_head_{leaf}"] = v.clone()                   # 末 stage + 下划线名
        elif k.startswith(("markov_head.", "confidence_head.")):
            out[f"mtp.{last}.{k}"] = v.clone()                              # 末 stage
        else:
            out[k] = v.clone()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    mc = load_file(os.path.join(args.src, "model.safetensors"))
    sv = to_serving(mc)

    ref = {}
    for f in sorted(glob.glob(os.path.join(PRETRAIN, "*.safetensors"))):
        ref.update({k: tuple(v.shape) for k, v in load_file(f).items()})
    bad = [(k, tuple(sv[k].shape), ref[k]) for k in sv if k in ref and tuple(sv[k].shape) != ref[k]]
    missing = [k for k in ref if k not in sv]          # 覆盖率断言(旧版缺失)
    extra = [k for k in sv if k not in ref]
    assert not bad, f"形状不符: {bad[:3]}"
    assert not missing, f"缺失预训练键 {len(missing)}: {missing[:6]}"
    # 允许的额外键:训练新增的 confidence_head.proj.bias(末 stage)
    last = max(int(m.group(1)) for k in mc if (m := re.match(r"layers\.(\d+)\.", k)))
    allowed_extra = {f"mtp.{last}.confidence_head.proj.bias"}
    bad_extra = [k for k in extra if k not in allowed_extra]
    assert not bad_extra, f"意外多出键 {len(bad_extra)}: {bad_extra[:6]}"
    print(f"[conv] oracle 通过:覆盖预训练 {len(ref)} 键,形状 0 不符,缺失 0,多出仅 {sorted(extra)}")

    os.makedirs(args.dst, exist_ok=True)
    save_file(sv, os.path.join(args.dst, "model.safetensors"))
    for f in os.listdir(PRETRAIN):
        if f.endswith((".json", ".txt", ".model")) and "safetensors" not in f:
            shutil.copy(os.path.join(PRETRAIN, f), os.path.join(args.dst, f))
    print(f"[conv] DONE {args.dst}: {len(sv)} 键")

if __name__ == "__main__":
    main()
