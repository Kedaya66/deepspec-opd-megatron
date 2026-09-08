"""TP=2 正确性验证:切开的注意力必须和不切时输出一致(纯 CPU,gloo)。

之前 demo 版在 TP=2 直接崩,根因是低秩瓶颈(q_lora_rank)被 ColumnParallelLinear
切了、而后面紧跟的 RMSNorm 要完整维度。现在的设计:
  * q_lora_rank / head_dim 复制,不切
  * 只切 head 维(wq_b / attn_sink),wo_a 按 group 切,wo_b RowParallel 收尾
本测试同时验证"切分语义正确",而不只是"不崩"。

    docker exec pytorch-2606 bash -lc \
        'cd /shenlb/zwf-spec/train-v4/deepspec-opd && \
         torchrun --nproc_per_node=2 --master_port=29733 tests/test_mcore_tp.py'
"""

import os
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dist.init_process_group(backend="gloo",
                        rank=int(os.environ["RANK"]),
                        world_size=int(os.environ["WORLD_SIZE"]))
TP = int(os.environ["WORLD_SIZE"])
RANK = dist.get_rank()

from megatron.core import parallel_state                                    # noqa: E402

parallel_state.initialize_model_parallel(tensor_model_parallel_size=TP,
                                         pipeline_model_parallel_size=1)
# 两个 rank 必须建出同一份 HF 参考权重
torch.manual_seed(1234)

from deepspec.modeling.dspark.deepseek_v4.layers_match_v1 import (          # noqa: E402
    DeepSeekV4DSparkAttentionMatchV1,
)
from deepspec.modeling.dspark.deepseek_v4.megatron.mcore_config import (    # noqa: E402
    DSparkParallelOptions, build_mcore_config,
)
from deepspec.modeling.dspark.deepseek_v4.megatron.mcore_layers import (    # noqa: E402
    McoreDSparkAttention,
)
from deepspec.modeling.dspark.deepseek_v4.megatron.convert_weights import (  # noqa: E402
    convert_layer_state_dict,
)

HID, HEADS, HEAD_DIM, ROPE_DIM = 64, 4, 16, 8
Q_LORA, O_LORA, GROUPS = 24, 12, 2
B, S, C = 2, 5, 7

hf_cfg = SimpleNamespace(
    hidden_size=HID, num_attention_heads=HEADS, head_dim=HEAD_DIM,
    qk_rope_head_dim=ROPE_DIM, q_lora_rank=Q_LORA, o_lora_rank=O_LORA,
    o_groups=GROUPS, rms_norm_eps=1e-6, attention_bias=False,
    attention_dropout=0.0, num_hidden_layers=1,
    moe_intermediate_size=32, n_routed_experts=4, n_shared_experts=1,
    num_experts_per_tok=2, scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5, swiglu_limit=0.0, vocab_size=128,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
)
opts = DSparkParallelOptions(tensor_model_parallel_size=TP, bf16=False,
                             moe_grouped_gemm=False, recompute_moe_layer=False)
cfg = build_mcore_config(hf_cfg, opts)
cfg.use_cpu_initialization = True
cfg.params_dtype = torch.float32
cfg.bf16 = False

hf_attn = DeepSeekV4DSparkAttentionMatchV1(hf_cfg, layer_idx=0).eval()
mc_attn = McoreDSparkAttention(cfg).eval()

sd = {f"self_attn.{k}": v for k, v in hf_attn.state_dict().items()}
mc_sd = convert_layer_state_dict(
    sd, num_heads=HEADS, head_dim=HEAD_DIM, n_groups=GROUPS, o_lora_rank=O_LORA,
    num_experts=0, tp_rank=RANK, tp_size=TP,
)
attn_sd = {k[len("self_attn."):]: v for k, v in mc_sd.items()}
missing, unexpected = mc_attn.load_state_dict(attn_sd, strict=False)
assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"

# 两个 rank 用同一份输入
torch.manual_seed(7)
x = torch.randn(B, S, HID)
tgt = torch.randn(B, C, HID)
inv = 1.0 / (10000 ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
f = torch.outer(torch.arange(C + S).float(), inv)
e = torch.cat([f, f], dim=-1)
cos, sin = e.cos().unsqueeze(0).expand(B, -1, -1), e.sin().unsqueeze(0).expand(B, -1, -1)

with torch.no_grad():
    ref, _ = hf_attn(hidden_states=x, target_hidden_states=tgt,
                     position_embeddings=(cos, sin), attention_mask=None)
    got = mc_attn(x, tgt, cos, sin, None)

d = (ref - got).abs().max().item()
# 两个 rank 的输出应完全一致(RowParallelLinear 已 all-reduce)
buf = got.clone()
dist.all_reduce(buf, op=dist.ReduceOp.MAX)
cross = (buf - got).abs().max().item()

if RANK == 0:
    print("=" * 78)
    print(f"TP={TP}  本 rank 持有 head={mc_attn.num_heads}/{HEADS} "
          f"group={mc_attn.n_groups}/{GROUPS}")
    print(f"  切分参数形状: wq_b={tuple(mc_attn.wq_b.weight.shape)} "
          f"wo_a={tuple(mc_attn.wo_a.shape)} "
          f"wo_b={tuple(mc_attn.wo_b.weight.shape)} "
          f"attn_sink={tuple(mc_attn.attn_sink.shape)}")
    print(f"  复制参数形状: wq_a={tuple(mc_attn.wq_a.weight.shape)} "
          f"wkv={tuple(mc_attn.wkv.weight.shape)}")
    ok = d <= 2e-5 and cross <= 1e-6
    print(f"  {'PASS' if d <= 2e-5 else 'FAIL'}  vs HF 参考(未切):  max|Δ|={d:.3e}")
    print(f"  {'PASS' if cross <= 1e-6 else 'FAIL'}  两 rank 输出一致:  max|Δ|={cross:.3e}")
    print("=" * 78)
    print("TP 切分语义正确" if ok else "TP 切分有问题")

parallel_state.destroy_model_parallel()
dist.destroy_process_group()
sys.exit(0 if (d <= 2e-5 and cross <= 1e-6) else 1)
