"""真实规模 DSpark draft 的 megatron 后端显存/吞吐测量。

真实维度(取自 DeepSeek-V4-Flash-DSpark-bf16/config.json):
  hidden=4096  heads=64  head_dim=512  q_lora=1024  o_lora=1024  o_groups=8
  moe_inter=2048  n_routed_experts=256  topk=6  n_shared=1   draft 层数=3

    torchrun --nproc_per_node=4 bench_ep.py --ep 4 --te
    torchrun --nproc_per_node=1 bench_ep.py --ep 1 --te --no-optim   # 对照
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--ep", type=int, default=4)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--te", action="store_true", help="TE 后端(grouped GEMM 必需)")
parser.add_argument("--fp8", default=None, choices=[None, "hybrid", "e4m3"])
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--experts", type=int, default=256)
parser.add_argument("--ctx", type=int, default=4096, help="target 上下文长度 C")
parser.add_argument("--anchors", type=int, default=128)
parser.add_argument("--block", type=int, default=5)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--no-optim", action="store_true")
parser.add_argument("--no-recompute", action="store_true")
args = parser.parse_args()

import torch
import torch.distributed as dist

sys.path.insert(0, "/shenlb/zwf-spec/train-v4/deepspec-opd")

LOCAL = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(LOCAL)
dist.init_process_group(backend="nccl",
                        rank=int(os.environ.get("RANK", 0)),
                        world_size=int(os.environ.get("WORLD_SIZE", 1)))
RANK = dist.get_rank()
WORLD = dist.get_world_size()

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module

from deepspec.modeling.dspark.deepseek_v4.megatron.mcore_config import (
    DSparkParallelOptions, build_mcore_config,
)

parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=args.tp,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=args.ep,
)
model_parallel_cuda_manual_seed(1234)

from deepspec.modeling.dspark.deepseek_v4.megatron.mcore_modeling import (
    build_dspark_layer_spec,
)

HID, HEADS, HEAD_DIM, ROPE = 4096, 64, 512, 64
S = args.anchors * args.block          # draft token 数
C = args.ctx
B = 1

hf_cfg = SimpleNamespace(
    hidden_size=HID, num_attention_heads=HEADS, head_dim=HEAD_DIM,
    qk_rope_head_dim=ROPE, q_lora_rank=1024, o_lora_rank=1024, o_groups=8,
    rms_norm_eps=1e-6, attention_bias=False, attention_dropout=0.0,
    num_hidden_layers=args.layers, moe_intermediate_size=2048,
    n_routed_experts=args.experts, n_shared_experts=1, num_experts_per_tok=6,
    scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=10.0,
    norm_topk_prob=True, vocab_size=129280,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
)
opts = DSparkParallelOptions(
    tensor_model_parallel_size=args.tp,
    expert_model_parallel_size=args.ep,
    moe_grouped_gemm=args.te,
    moe_token_dispatcher_type="alltoall" if args.ep > 1 else "allgather",
    bf16=True, fp8=args.fp8, use_transformer_engine=args.te,
    recompute_moe_layer=not args.no_recompute,
)
cfg = build_mcore_config(hf_cfg, opts)
cfg.use_cpu_initialization = False

pgs = ProcessGroupCollection.use_mpu_process_groups()
spec = build_dspark_layer_spec(cfg, opts)

torch.cuda.reset_peak_memory_stats()
layers = torch.nn.ModuleList([
    build_module(spec, config=cfg, layer_number=i + 1, pg_collection=pgs)
    for i in range(args.layers)
]).cuda()
mem_params = torch.cuda.memory_allocated() / 2**30
n_local = sum(p.numel() for p in layers.parameters())

opt = None
if not args.no_optim:
    opt = torch.optim.Adam(layers.parameters(), lr=1e-5)

# rope 表 + 输入
inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device="cuda").float() / ROPE))
f = torch.outer(torch.arange(C + S, device="cuda").float(), inv)
e = torch.cat([f, f], dim=-1)
cos = e.cos().unsqueeze(0).expand(B, -1, -1).bfloat16()
sin = e.sin().unsqueeze(0).expand(B, -1, -1).bfloat16()
streams = torch.randn(B, S, 4, HID, device="cuda", dtype=torch.bfloat16)
tgt = torch.randn(B, C, HID, device="cuda", dtype=torch.bfloat16)


def one_step():
    h = streams
    for lyr in layers:
        h, _ = lyr(h, target_hidden_states=tgt, cos=cos, sin=sin)
    loss = h.float().pow(2).mean()
    loss.backward()
    if opt is not None:
        opt.step()
        opt.zero_grad(set_to_none=True)
    else:
        for p in layers.parameters():
            p.grad = None
    return loss.item()


times = []
for it in range(args.steps):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = one_step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if it >= 2:                      # 前两步是 warmup
        times.append(dt)
    if RANK == 0:
        print(f"  step {it}: {dt*1000:.0f} ms  loss={loss:.4f}"
              f"  峰值={torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

peak = torch.cuda.max_memory_allocated() / 2**30
reserved = torch.cuda.max_memory_reserved() / 2**30
avg = sum(times) / max(len(times), 1)

# 收集各 rank 的峰值
stat = torch.tensor([peak, reserved, n_local / 1e9, avg], device="cuda")
allstat = [torch.zeros_like(stat) for _ in range(WORLD)]
dist.all_gather(allstat, stat)

if RANK == 0:
    expert_layout = type(layers[0].mlp.experts).__name__
    print("=" * 82)
    print(f"配置: EP={args.ep} TP={args.tp} 层数={args.layers} 专家={args.experts} "
          f"后端={'TE' if args.te else 'local'} fp8={args.fp8} "
          f"重算={'关' if args.no_recompute else '开'} 优化器={'无' if args.no_optim else 'Adam'}")
    print(f"形状: B={B} draft_S={S}(anchors={args.anchors}x block={args.block}) ctx_C={C}")
    print(f"MoE experts 实际类型: {expert_layout}")
    print(f"attention scores 张量: [1,{HEADS},{S},{C+S}] = "
          f"{HEADS*S*(C+S)*2/2**30:.2f} GiB (bf16, 每层)")
    print("-" * 82)
    print(f"{'rank':>4} {'本地参数(B)':>12} {'峰值(GiB)':>11} {'reserved(GiB)':>14} {'步时(ms)':>10}")
    for r, s in enumerate(allstat):
        print(f"{r:>4} {s[2].item():>12.3f} {s[0].item():>11.1f} {s[1].item():>14.1f} "
              f"{s[3].item()*1000:>10.0f}")
    tot = sum(s[2].item() for s in allstat)
    print("-" * 82)
    print(f"全局参数量: {tot:.2f} B (bf16 权重 {tot*2:.1f} GB)")
    print(f"平均步时: {avg*1000:.0f} ms  ->  {S/avg:.0f} draft-token/s")
    print("=" * 82)

parallel_state.destroy_model_parallel()
dist.destroy_process_group()
