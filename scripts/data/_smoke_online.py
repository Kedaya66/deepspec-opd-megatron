"""在线训练链路轻量冒烟(单 GPU,小 draft,不 dist/不 FSDP/不加载 target)。"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")  # 服务占 0-3
import sys
import copy

import torch

ROOT = "/shenlb/zwf-spec/train-v1/deepspec"
sys.path.insert(0, ROOT)

import torch.distributed as dist
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=0, world_size=1)

from deepspec.utils import load_config
from deepspec.utils.config import to_config_node

cfg = load_config(f"{ROOT}/config/dspark/dspark_dskv4-flash.py")
mp = str(cfg.model.target_model_name_or_path)

# --- V4-Flash 官方 encode 渲染覆盖(与在线 trainer 的 _setup_data_encoding 一致) ---
enc = os.path.join(mp, "encoding")
sys.path.insert(0, enc)
from encoding_dsv4 import encode_messages
from deepspec.data import parser as P

def _flat(c):
    return "".join(b.get("text", "") for b in c if isinstance(b, dict)) if isinstance(c, list) else (c or "")

P.set_render_override(lambda tok, m, *, add_generation_prompt, enable_thinking=None:
                      encode_messages([dict(x, content=_flat(x.get("content"))) for x in m],
                                      thinking_mode="chat", add_default_bos_token=True))

from transformers import AutoTokenizer, AutoConfig
tok = AutoTokenizer.from_pretrained(mp)
target_config = AutoConfig.from_pretrained(mp)

# --- 建「小」draft:num_draft_layers=1, num_anchors=64(省显存/加速),随机 embed/head ---
from deepspec.modeling.dspark.deepseek_v4 import build_flash_draft_config, DeepSeekV4FlashDSparkModel
ma = dict(cfg.model)
ma["num_draft_layers"] = 1
ma["num_anchors"] = 64
model_args = to_config_node(ma)
draft_config = build_flash_draft_config(target_config=target_config, model_args=model_args)
model = DeepSeekV4FlashDSparkModel(draft_config).to("cuda", torch.bfloat16).train()
n_par = sum(p.numel() for p in model.parameters())
print(f"[smoke] draft 建好: {n_par/1e9:.1f}B 参数, num_draft_layers=1, num_anchors=64")

# --- 数据:取 1 条,截到 2048 token 加速 ---
from deepspec.data import ConversationCollator
from deepspec.data.jsonl_dataset import JsonLineDataset
ds = JsonLineDataset(data_paths=list(cfg.data.train_data_path))
coll = ConversationCollator(tokenizer=tok, chat_template=cfg.data.chat_template,
                            max_length=cfg.data.max_length, min_loss_tokens=0)
batch = coll([ds[0]])
SEQ = 2048
batch = {k: v[:, -SEQ:].contiguous() for k, v in batch.items()}
print(f"[smoke] batch: input_ids={tuple(batch['input_ids'].shape)} "
      f"loss_mask.sum={int(batch['loss_mask'].sum())}")
batch = {k: v.to("cuda") for k, v in batch.items()}

# --- 在线取特征(tp4 服务)---
from deepspec.trainer.dspark_online_trainer import fetch_target_features
th, tl = fetch_target_features(
    endpoint="http://127.0.0.1:30000",
    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
    target_layer_ids=list(cfg.model.target_layer_ids),
)
print(f"[smoke] 取到特征: target_hidden={tuple(th.shape)} last={tuple(tl.shape)} "
      f"(期望 [1,{SEQ},{3*4096}] / [1,{SEQ},4096])")
th = th.to("cuda", torch.bfloat16); tl = tl.to("cuda", torch.bfloat16)

# --- forward + loss + backward ---
from deepspec.modeling.dspark.loss import compute_dspark_loss
out = model(input_ids=batch["input_ids"], target_hidden_states=th,
            loss_mask=batch["loss_mask"], target_last_hidden_states=tl)
loss = compute_dspark_loss(outputs=out,
                           loss_decay_gamma=cfg.model.loss_decay_gamma,
                           ce_loss_alpha=float(cfg.model.ce_loss_alpha),
                           l1_loss_alpha=float(cfg.model.l1_loss_alpha),
                           confidence_head_alpha=float(cfg.model.confidence_head_alpha))
loss.backward()
n_grad = sum(1 for p in model.parameters() if p.grad is not None)
print(f"[smoke] loss={loss.item():.4f} finite={bool(torch.isfinite(loss))} "
      f"| 有梯度的参数张量数={n_grad}")
print("[smoke] ✅ 在线链路端到端跑通" if torch.isfinite(loss) and n_grad > 0 else "[smoke] ❌ 有问题")
