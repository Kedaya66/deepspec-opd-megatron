"""临时冒烟配置:快速验证在线训练整条链路能走到 checkpoint 落盘。
小 draft(num_draft_layers=1, 不加载 39GB 预训练)、关 compile、1 步即存盘。"""
import os

from deepspec.trainer import DeepSeekV4FlashDSparkOnlineTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, DSK_V4_FLASH

project_name = "deepspec"
exp_name = "SMOKE_online_v4flash"
seed = 42

PRETRAINED_DSPARK = "/shenlb/zwf-spec/train/model/DeepSeek-V4-Flash-DSpark-bf16"

model = dict(
    target_model_name_or_path=DSK_V4_FLASH,
    block_size=5,
    num_draft_layers=1,          # 冒烟:小 draft
    target_layer_ids=[40, 41, 42],
    mask_token_id=128799,
    num_anchors=64,              # 冒烟:少 anchor
    markov_rank=256,
    markov_head_type='vanilla',
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
    pretrained_dspark_path=PRETRAINED_DSPARK,  # 仍用它初始化 embed/head
    load_pretrained_weights=False,             # 冒烟:不加载 39GB draft 权重(随机初始化)
)

train = dict(
    trainer_cls=DeepSeekV4FlashDSparkOnlineTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=4,         # 冒烟:4 卡 × 1,grad_accum=1
    num_train_epochs=1,
    max_train_steps=1,           # 冒烟:1 个 optimizer step
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    torch_compile=False,         # 冒烟:关 compile 加速启动
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=1,       # 冒烟:第 1 步就存盘
)

data = dict(
    target_cache_path=None,
    train_data_path=[
        "/shenlb/zwf-spec/dataset/dataset-v1/主agent输入_医疗助手_正式环境_perfectblend_openai_conv_regen.jsonl",
    ],
    chat_template="v4-flash",
    max_length=100000,           # 不截断(样本<25k),保留末尾 assistant 使 loss_mask>0
    num_workers=2,
    min_loss_tokens=0,
)

online = dict(
    sglang_server_address=["127.0.0.1:30000"],
    request_timeout=1200.0,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg
