"""match-v1 模型 debug 脚本:前向逐模块检查 + 损失 + 反向梯度体检。

用法(默认 tiny 配置,CPU 秒级):
    python scripts/debug_match_v1.py                        # tiny 冒烟
    python scripts/debug_match_v1.py --verbose              # 打印每个子模块
    python scripts/debug_match_v1.py --anomaly              # NaN 溯源(autograd 异常检测)
    python scripts/debug_match_v1.py --mode real \
        --target-config /shenlb/zwf-spec/train/model/DeepSeek-V4-Flash-DSpark-bf16 \
        [--load 同上路径]                                    # 真配置(20B,吃内存,可选载官方权重)

检查项:
  1. 前向:每个关键模块的输出 shape/均值/方差/绝对值峰值 + NaN/Inf 报警;
  2. 输出:DSparkForwardOutput 六个字段的形状与统计,eval_mask 有效率;
  3. 损失:优先走真实 compute_dspark_loss(自动起单进程 gloo),失败回退简化损失;
  4. 反向:按模块分组的梯度范数、无梯度参数清单、NaN 梯度报警;
  5. 确定性:同种子跑两遍前向,输出应逐位一致(抓状态污染/非确定算子)。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from transformers import PretrainedConfig


def tstats(t):
    if not torch.is_tensor(t): return str(type(t).__name__)
    x = t.detach().float()
    s = f"{str(tuple(t.shape)):<22} μ={x.mean():+.3e} σ={x.std():.3e} |max|={x.abs().max():.3e}"
    if torch.isnan(x).any(): s += "  ⚠⚠ NaN"
    if torch.isinf(x).any(): s += "  ⚠⚠ Inf"
    return s


def build_tiny_config():
    return PretrainedConfig(
        vocab_size=97, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, head_dim=32, qk_rope_head_dim=8,
        q_lora_rank=16, o_lora_rank=16, o_groups=2,
        rms_norm_eps=1e-6, rope_theta=10000.0,
        n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
        target_layer_ids=[0, 1], block_size=3, mask_token_id=96, num_anchors=4,
        markov_rank=8, markov_head_type="vanilla",
        enable_confidence_head=True, confidence_head_with_markov=True,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    )


def build_real_config(target_path):
    from deepspec.modeling.dspark.deepseek_v4.load_pretrained import _AttrDict, load_target_config
    from deepspec.modeling.dspark.deepseek_v4.config import build_flash_draft_config
    target_config = load_target_config(target_path)
    model_args = _AttrDict(
        num_draft_layers=3, target_layer_ids=[40, 41, 42], block_size=5,
        mask_token_id=128799, num_anchors=8, markov_rank=256, markov_head_type="vanilla",
        confidence_head_alpha=1.0, confidence_head_with_markov=True,
    )
    return build_flash_draft_config(target_config=target_config, model_args=model_args)


def hook_modules(model, verbose=False):
    """给关键模块挂前向钩子,打印输出统计。返回 remove 函数。"""
    keys = ("embed_tokens", "fc", "hidden_norm", "self_attn", "attn_hc", "ffn_hc",
            "mlp", "hc_head", "norm", "markov_head", "confidence_head")
    handles = []
    def make_hook(name):
        def hook(mod, args, out):
            o = out
            if isinstance(o, tuple):
                tensors = [t for t in o if torch.is_tensor(t)]
                o = tensors[-1] if tensors else None   # HC→collapsed, attn→输出
            print(f"    {name:<42} {tstats(o)}")
        return hook
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if verbose and name and len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(make_hook(name)))
        elif not verbose and (leaf in keys or name in keys):
            handles.append(mod.register_forward_hook(make_hook(name)))
    return lambda: [h.remove() for h in handles]


def try_real_loss(out):
    """优先真实损失(需要 dist);失败回退简化损失。返回 (loss, 用的哪条路)。"""
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29517")
            dist.init_process_group("gloo", rank=0, world_size=1)
        from deepspec.modeling.dspark.loss import compute_dspark_loss
        loss = compute_dspark_loss(outputs=out, loss_decay_gamma=4.0,
                                   ce_loss_alpha=0.1, l1_loss_alpha=0.9,
                                   confidence_head_alpha=1.0)
        return loss, "compute_dspark_loss(真实)"
    except Exception as e:
        lg = out.draft_logits.float()
        loss = torch.nn.functional.cross_entropy(
            lg.flatten(0, 2), out.target_ids.flatten(), reduction="none")
        loss = (loss * out.eval_mask.flatten().float()).sum() / out.eval_mask.sum().clamp(min=1)
        if out.confidence_pred is not None:
            loss = loss + out.confidence_pred.float().pow(2).mean() * 0.1
        return loss, f"简化损失(真实损失失败: {type(e).__name__}: {e})"


def grad_report(model):
    groups = {}
    none_grads, nan_grads = [], []
    for name, p in model.named_parameters():
        top = ".".join(name.split(".")[:2]) if name.startswith("layers") else name.split(".")[0]
        g = groups.setdefault(top, {"n": 0, "with_grad": 0, "norm2": 0.0})
        g["n"] += 1
        if p.grad is None:
            none_grads.append(name); continue
        g["with_grad"] += 1
        gn = p.grad.detach().float().norm().item()
        g["norm2"] += gn * gn
        if torch.isnan(p.grad).any(): nan_grads.append(name)
    print(f"\n== 梯度体检(按模块分组) ==")
    for k in sorted(groups):
        g = groups[k]
        print(f"    {k:<28} 参数 {g['with_grad']}/{g['n']} 有梯度   |grad| = {g['norm2'] ** 0.5:.3e}")
    if none_grads:
        print(f"    ⚠ 无梯度参数 {len(none_grads)} 个(冻结的 embed/lm_head 属正常): "
              + ", ".join(none_grads[:6]) + (" ..." if len(none_grads) > 6 else ""))
    if nan_grads:
        print(f"    ⚠⚠ NaN 梯度: {nan_grads[:8]}")
    return nan_grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tiny", "real"], default="tiny")
    ap.add_argument("--target-config", default="/shenlb/zwf-spec/train/model/DeepSeek-V4-Flash-DSpark-bf16")
    ap.add_argument("--load", default=None, help="官方 DSpark ckpt 路径(real 模式,载入权重;吃 ~50GB 内存)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--prompt-len", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="每个叶子模块都打印")
    ap.add_argument("--anomaly", action="store_true", help="autograd 异常检测(NaN 溯源,慢)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.anomaly: torch.autograd.set_detect_anomaly(True)

    from deepspec.modeling.dspark.deepseek_v4.modeling_match_v1 import DeepSeekV4FlashDSparkModelMatchV1

    print(f"== 构建模型({args.mode}) ==")
    t0 = time.time()
    cfg = build_tiny_config() if args.mode == "tiny" else build_real_config(args.target_config)
    model = DeepSeekV4FlashDSparkModelMatchV1(cfg).float()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    参数量 {n_params/1e6:.1f}M · hc_mult={model.hc_mult} · block_size={model.block_size} "
          f"· 层数 {len(model.layers)} · 构建 {time.time()-t0:.1f}s")

    if args.load:
        from deepspec.modeling.dspark.deepseek_v4.load_pretrained_match_v1 import apply_pretrained_match_v1_weights
        print("== 载入官方权重(含 rope 置换/hc/sink) ==")
        apply_pretrained_match_v1_weights(model, args.load)
        model.initialize_embeddings_and_head(embed_tokens=model.embed_tokens, lm_head=model.lm_head)  # 冻结示意

    B, T, P = args.batch, args.seq_len, args.prompt_len
    D, V = cfg.hidden_size, cfg.vocab_size
    ids = torch.randint(0, V - 1, (B, T))
    loss_mask = torch.ones(B, T); loss_mask[:, :P] = 0
    tgt_hidden = torch.randn(B, T, len(cfg.target_layer_ids) * D)
    tgt_last = torch.randn(B, T, D)

    print(f"\n== 前向(B={B}, T={T}, prompt={P}, anchors={model.num_anchors}, K={model.block_size}) ==")
    unhook = hook_modules(model, args.verbose)
    torch.manual_seed(args.seed + 1)   # 固定 anchor 采样
    t0 = time.time()
    out = model(input_ids=ids, target_hidden_states=tgt_hidden,
                loss_mask=loss_mask, target_last_hidden_states=tgt_last)
    unhook()
    print(f"    前向耗时 {time.time()-t0:.2f}s")

    print(f"\n== 输出(DSparkForwardOutput) ==")
    print(f"    draft_logits          {tstats(out.draft_logits)}")
    print(f"    target_ids            {str(tuple(out.target_ids.shape)):<22} 样例 {out.target_ids[0,0].tolist()}")
    print(f"    aligned_target_logits {tstats(out.aligned_target_logits)}")
    print(f"    confidence_pred       {tstats(out.confidence_pred)}")
    em = out.eval_mask
    print(f"    eval_mask             {str(tuple(em.shape)):<22} 有效槽 {int(em.sum())}/{em.numel()}"
          f" · 有效块 {int(out.block_keep_mask.sum())}/{out.block_keep_mask.numel()}")
    for k in range(model.block_size):
        print(f"      槽{k}: 有效 {int(em[..., k].sum())}")

    print(f"\n== 损失 + 反向 ==")
    loss, path = try_real_loss(out)
    print(f"    损失路径: {path}")
    print(f"    loss = {loss.item():.4f}")
    t0 = time.time()
    loss.backward()
    print(f"    反向耗时 {time.time()-t0:.2f}s")
    nan_grads = grad_report(model)

    print(f"\n== 确定性:同种子重跑前向 ==")
    model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed + 1)
    out2 = model(input_ids=ids, target_hidden_states=tgt_hidden,
                 loss_mask=loss_mask, target_last_hidden_states=tgt_last)
    same = torch.equal(out.draft_logits, out2.draft_logits)
    print(f"    draft_logits 两次逐位一致: {'PASS ✓' if same else 'FAIL ✗(检查非确定算子/状态污染)'}")

    bad = (not same) or bool(nan_grads) or torch.isnan(out.draft_logits).any()
    print(f"\n== 总结: {'⚠ 有问题,往上翻 ⚠ 标记' if bad else '全部正常 ✓'} ==")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
