"""OOM 压测配置:CPU offload + 最长真实样本(58K token)×4,单步跑 forward+backward+step。
load_pretrained_weights=False 只为省 39GB 加载(显存占用与加载后一致)。"""
import os

from deepspec.trainer import DeepSeekV4FlashDSparkOnlineTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, DSK_V4_FLASH

project_name = "deepspec"
exp_name = "STRESS_offload_58k"
seed = 42
PRETRAINED_DSPARK = "/shenlb/zwf-spec/train/model/DeepSeek-V4-Flash-DSpark-bf16"

model = dict(
    target_model_name_or_path=DSK_V4_FLASH,
    block_size=5,
    num_draft_layers=3,                # 全量 20B,压测最坏
    target_layer_ids=[40, 41, 42],
    mask_token_id=128799,
    num_anchors=256,
    markov_rank=256,
    markov_head_type='vanilla',
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
    pretrained_dspark_path=PRETRAINED_DSPARK,
    load_pretrained_weights=False,     # 省 39GB 加载(显存与加载后相同)
    gradient_checkpointing=True,       # 模型若不支持会自动跳过
)

train = dict(
    trainer_cls=DeepSeekV4FlashDSparkOnlineTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=4,               # 4 卡 ×1,grad_accum=1 → 每卡 1 条 58K
    num_train_epochs=1,
    max_train_steps=1,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    cpu_offload=True,                  # ← 本次压测重点
    torch_compile=False,
)

logging = dict(logging_steps=1, checkpointing_steps=1)

data = dict(
    target_cache_path=None,
    train_data_path=["/shenlb/zwf-spec/dataset/dataset-v1/_stress_longest_x4.jsonl"],
    chat_template="v4-flash",
    max_length=50000,                 # 不截断,保留全 58K
    num_workers=1,
    min_loss_tokens=0,
)

online = dict(
    sglang_server_address=["127.0.0.1:30000"],
    request_timeout=1800.0,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, str(cfg['project_name']), str(cfg["exp_name"]))
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, str(cfg['project_name']), str(cfg["exp_name"]))
    cfg["logging"] = logging_cfg
    return cfg
