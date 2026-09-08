import os

from deepspec.trainer import DeepSeekV4FlashDSparkOnlineTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, QWEN_3_4B, DSK_V4_FLASH

project_name = "deepspec"
exp_name = "dspark_opd_smoke"
seed = 42

# 预训练 DSpark draft(bf16),num_draft_layers=3 对应其 3 个 mtp block
PRETRAINED_DSPARK = "/shenlb/zwf-spec/train/model/DeepSeek-V4-Flash-DSpark-bf16"

model = dict(
    target_model_name_or_path=DSK_V4_FLASH,
    block_size=5,
    num_draft_layers=3,
    target_layer_ids=[40, 41, 42],

    mask_token_id=128799,  # = dspark_noise_token_id (<｜place holder no 799｜>)
    num_anchors=128,  # match_v1 eager+mHC 显存重;128 时 scores ~7.4GB,较 256 省 ~7GB

    ## markov head
    markov_rank=256,
    markov_head_type='vanilla',

    ## confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,

    ## 预训练权重(热启动 draft);置 None 或 load_pretrained_weights=False 则随机初始化
    pretrained_dspark_path=PRETRAINED_DSPARK,
    load_pretrained_weights=True,
    gradient_checkpointing=True,  # 256-expert draft 反向省显存
)

train = dict(
    trainer_cls=DeepSeekV4FlashDSparkOnlineTrainer,
    lr=1.0e-5,                       # 6e-4→1e-5:微调预训练draft,照搬gentle(2.73)
    warmup_ratio=0.05,               # 对齐gentle;110步→~5步爬升
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=16,
    num_train_epochs=2,
    max_train_steps=2,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    cpu_offload=False,  # VERL FSDP2 训练默认:参数/梯度/优化器分片留在 GPU
    torch_compile=True,
)

logging = dict(
    logging_steps=1,
    # 每 global step = global_batch_size(512) 个样本;6 step ≈ 3072 样本 ≈ 每 3000 样本存一次
    checkpointing_steps=1000000,  # 每步存最新,方便 resume
    save_total_limit=2,  # 只留当前(权重+优化器)+上一step(strip-before剥成仅权重)
)

data = dict(
    # 在线训练:直接给对话 JSONL(无需离线 target cache)
    target_cache_path=None,
    train_data_path=[
        "/shenlb/zwf-spec/dataset/dataset-v1/主agent输入_医疗助手_正式环境_perfectblend_openai_conv_regen.jsonl",
    ],
    chat_template="v4-flash",
    max_length=45000,
    num_workers=4,
    # 0 保证每 batch 非空(避免 CUDAPrefetcher 遇 None);>0 会丢弃 loss token 过少的样本
    min_loss_tokens=0,
)

# 在线取特征:指向已部署的 SGLang 服务(带 SGLANG_DSPARK_EXPORT_LAYERS=40,41,42)
online = dict(
    sglang_server_address=["127.0.0.1:30000"],  # tp4 服务
    request_timeout=1200.0,
    ## OPD(on-policy distillation)方案一:target rollout 自产轨迹 + 同趟特征
    opd_rollout=True,
    opd_max_new_tokens=512,   # 每条 rollout 生成上限(自然 EOS 提前停)
    opd_temperature=0.0,       # 对齐 serving(greedy);fp8+MoE 非确定自带轨迹多样性
    opd_top_p=1.0,
    opd_prefetch=8,
    opd_max_prefetch=16,  # 自适应窗口上限(SWA 池约束)      # 每 rank read-ahead 并发 rollout 数(4 rank 共 32 并发)
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join("/root/checkpoints", project_name, exp_name)  # overlay,绕开坏掉的 cubefs  # cubefs 37T,避开 /root 小配额
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg
