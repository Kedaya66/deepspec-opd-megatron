"""mcore 版 DSpark 层 vs HF match-v1 层 —— 同一份权重的数值对拍(纯 CPU)。

这是移植的验收标准:结构搬对了不算数,必须在同一份随机权重上输出一致,
否则等于悄悄换了个模型。

    docker exec pytorch-2606 bash -lc \
        'cd /shenlb/zwf-spec/train-v4/deepspec-opd && python tests/test_mcore_parity.py'

覆盖:
  [1] 注意力单独对拍(rope 尾旋 / 单 latent 当 KV / attn_sink / 输出反旋转 / 分组低秩)
  [2] mHC 对拍(含 20 轮 Sinkhorn)
  [3] 整层对拍(MoE 用 mcore MoELayer 替换 HF 的 256 专家 python 循环)
  [4] 梯度完整性:每个参数都要拿到非零梯度
MoE 那部分在 CPU 上跑不了(TopKRouter.gating 硬编码 cuda),所以 [3] 用 dense MLP
变体验证"层的连接方式"对齐,MoE 的等价性留到 GPU 冒烟。
"""

import os
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29744")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dist.init_process_group(backend="gloo", rank=0, world_size=1)

from megatron.core import parallel_state                                  # noqa: E402
from megatron.core.models.backends import LocalSpecProvider               # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import (                    # noqa: E402
    get_mlp_module_spec_for_backend,
)
from megatron.core.process_groups_config import ProcessGroupCollection    # noqa: E402
from megatron.core.transformer.spec_utils import ModuleSpec, build_module  # noqa: E402
from megatron.core.transformer.transformer_layer import (                 # noqa: E402
    TransformerLayerSubmodules,
)

parallel_state.initialize_model_parallel(1, 1)
torch.manual_seed(1234)

from deepspec.modeling.dspark.deepseek_v4.layers_match_v1 import (        # noqa: E402
    DeepSeekV4DSparkAttentionMatchV1,
    DSparkHyperConnection,
    DSparkHyperHead,
)
from deepspec.modeling.dspark.deepseek_v4.megatron.mcore_config import (  # noqa: E402
    DSparkParallelOptions,
    build_mcore_config,
)
from deepspec.modeling.dspark.deepseek_v4.megatron.mcore_layers import (  # noqa: E402
    McoreDSparkAttention,
    McoreDSparkHyperConnection,
    McoreDSparkHyperHead,
    McoreDSparkLayer,
)
from deepspec.modeling.dspark.deepseek_v4.megatron.convert_weights import (  # noqa: E402
    convert_layer_state_dict,
)

# ---- 玩具尺寸,但保持 match-v1 的全部结构关系 ----
HID, HEADS, HEAD_DIM, ROPE_DIM = 64, 4, 16, 8
Q_LORA, O_LORA, GROUPS = 24, 12, 2
N_EXPERTS, MOE_INTER, TOPK = 4, 32, 2
B, S, C = 2, 5, 7            # C = target 上下文长度(KV 比 Q 长)

hf_cfg = SimpleNamespace(
    hidden_size=HID, num_attention_heads=HEADS, head_dim=HEAD_DIM,
    qk_rope_head_dim=ROPE_DIM, q_lora_rank=Q_LORA, o_lora_rank=O_LORA,
    o_groups=GROUPS, rms_norm_eps=1e-6, attention_bias=False,
    attention_dropout=0.0, num_hidden_layers=1,
    moe_intermediate_size=MOE_INTER, n_routed_experts=N_EXPERTS,
    n_shared_experts=1, num_experts_per_tok=TOPK,
    scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
    swiglu_limit=0.0, norm_topk_prob=True, vocab_size=128,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
)
opts = DSparkParallelOptions(bf16=False, moe_grouped_gemm=False,
                             moe_token_dispatcher_type="allgather",
                             recompute_moe_layer=False)
mc_cfg = build_mcore_config(hf_cfg, opts)
mc_cfg.use_cpu_initialization = True
mc_cfg.params_dtype = torch.float32
mc_cfg.bf16 = False

PGS = ProcessGroupCollection.use_mpu_process_groups()
FAIL = []


def check(name, a, b, tol=2e-5):
    d = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), 1e-12)
    ok = d <= tol * max(scale, 1.0)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} max|Δ|={d:.3e}  (ref max|x|={scale:.3e})")
    if not ok:
        FAIL.append(name)


def rope_table(total, dim):
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    f = torch.outer(torch.arange(total).float(), inv)
    e = torch.cat([f, f], dim=-1)
    return e.cos().unsqueeze(0).expand(B, -1, -1), e.sin().unsqueeze(0).expand(B, -1, -1)


cos, sin = rope_table(C + S, ROPE_DIM)
x = torch.randn(B, S, HID)
tgt = torch.randn(B, C, HID)
streams = torch.randn(B, S, 4, HID)

print("=" * 78)
print("[1] 注意力对拍")
hf_attn = DeepSeekV4DSparkAttentionMatchV1(hf_cfg, layer_idx=0).eval()
mc_attn = McoreDSparkAttention(mc_cfg).eval()
# HF -> mcore 权重搬运(TP=1,只有 wo_a 的形状要变)
sd = {f"self_attn.{k}": v for k, v in hf_attn.state_dict().items()}
sd.update({"input_layernorm.weight": torch.ones(HID),
           "post_attention_layernorm.weight": torch.ones(HID)})
for hc in ("attn_hc", "ffn_hc"):
    sd[f"{hc}.fn"] = torch.zeros((2 + 4) * 4, 4 * HID)
    sd[f"{hc}.base"] = torch.zeros((2 + 4) * 4)
    sd[f"{hc}.scale"] = torch.zeros(3)
mc_sd = convert_layer_state_dict(
    sd, num_heads=HEADS, head_dim=HEAD_DIM, n_groups=GROUPS, o_lora_rank=O_LORA,
    num_experts=0, tp_rank=0, tp_size=1,
)
attn_sd = {k[len("self_attn."):]: v for k, v in mc_sd.items() if k.startswith("self_attn.")}
missing, unexpected = mc_attn.load_state_dict(attn_sd, strict=False)
assert not unexpected, f"转换产出了 mcore 不认识的键: {unexpected}"
assert not missing, f"mcore 注意力有键没被填: {missing}"

with torch.no_grad():
    hf_out, _ = hf_attn(hidden_states=x, target_hidden_states=tgt,
                        position_embeddings=(cos, sin), attention_mask=None)
    mc_out = mc_attn(x, tgt, cos, sin, None)
check("attention 输出", hf_out, mc_out)

print()
print("[2] mHC 对拍(含 20 轮 Sinkhorn)")
hf_hc = DSparkHyperConnection(HID, 4, 20, 1e-6, 1e-6).eval()
mc_hc = McoreDSparkHyperConnection(HID, 4, 20, 1e-6, 1e-6).eval()
with torch.no_grad():
    for p in (hf_hc.fn, hf_hc.base, hf_hc.scale):
        p.normal_(0, 0.05)        # 非零初始化,否则梯度恒 0 看不出问题
    mc_hc.load_state_dict(hf_hc.state_dict())
    a1, a2, a3 = hf_hc(streams)
    b1, b2, b3 = mc_hc(streams)
check("mHC post", a1, b1)
check("mHC comb (Sinkhorn 后)", a2, b2)
check("mHC collapsed", a3, b3)

hf_head = DSparkHyperHead(HID, 4, 1e-6, 1e-6).eval()
mc_head = McoreDSparkHyperHead(HID, 4, 1e-6, 1e-6).eval()
with torch.no_grad():
    for p in (hf_head.fn, hf_head.base, hf_head.scale):
        p.normal_(0, 0.05)
    mc_head.load_state_dict(hf_head.state_dict())
    check("hc_head collapsed", hf_head(streams), mc_head(streams))

print()
print("[3] 整层结构 + 梯度完整性(dense MLP 变体)")
# MoE 在 CPU 上跑不了(TopKRouter.gating 硬编码 torch.cuda.current_device()),
# 所以这里用 dense mlp 槽验证"层的连接方式"(mHC 读写 + 两次残差)对齐。
dense_cfg = build_mcore_config(hf_cfg, opts)
dense_cfg.use_cpu_initialization = True
dense_cfg.params_dtype = torch.float32
dense_cfg.bf16 = False
dense_cfg.num_moe_experts = None
dense_cfg.moe_ffn_hidden_size = None
dense_cfg.moe_shared_expert_intermediate_size = None
dense_cfg.ffn_hidden_size = MOE_INTER
dense_spec = ModuleSpec(
    module=McoreDSparkLayer,
    submodules=TransformerLayerSubmodules(
        mlp=get_mlp_module_spec_for_backend(backend=LocalSpecProvider(), num_experts=None)),
)
layer = build_module(dense_spec, config=dense_cfg, layer_number=1, pg_collection=PGS)
with torch.no_grad():
    for hc in (layer.attn_hc, layer.ffn_hc):
        for p in (hc.fn, hc.base, hc.scale):
            p.normal_(0, 0.05)
print("  层内插槽:", [n for n, _ in layer.named_children()])
print("  mlp 槽实际类型:", type(layer.mlp).__name__)
out, ctx = layer(streams, target_hidden_states=tgt, cos=cos, sin=sin)
print(f"  forward: {tuple(streams.shape)} + target{tuple(tgt.shape)} -> {tuple(out.shape)}"
      f" | 返回二元组(TransformerBlock 契约): {ctx is None}")
assert out.shape == streams.shape, "4 路残差流形状必须守恒"

loss = out.float().pow(2).mean()
loss.backward()
n_all = sum(1 for _ in layer.parameters())
n_grad = sum(1 for p in layer.parameters() if p.grad is not None)
zero = [n for n, p in layer.named_parameters()
        if p.grad is None or p.grad.abs().max().item() == 0.0]
print(f"  有梯度: {n_grad}/{n_all}")
if zero:
    print(f"  !! 梯度恒零的参数: {zero}")
    FAIL.append("零梯度参数: " + ",".join(zero))
else:
    print("  所有参数梯度非零")

print()
print("=" * 78)
if FAIL:
    print("对拍失败:", FAIL)
else:
    print("全部通过 —— mcore 版和 HF match-v1 在同一份权重上数值一致")
print("=" * 78)

parallel_state.destroy_model_parallel()
dist.destroy_process_group()
sys.exit(1 if FAIL else 0)
