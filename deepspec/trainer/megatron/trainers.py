"""Megatron 后端的算法 trainer 组合类。

组合规则:算法类在前、MegatronBaseTrainer 在后。
MRO 形如 [Megatron<Algo>Trainer, <Algo>Trainer(...), MegatronBaseTrainer, BaseTrainer],
于是:
- 算法侧覆写(_build_draft_model / build_models / run_batch / 在线 mixin 的
  数据集与 dataloader)优先生效;
- 基础设施(__init__ / train / checkpoint / clean_up / 默认 dataloader)落到
  MegatronBaseTrainer,因为 BaseTrainer 排在它后面。

注意:DSparkOnlineTrainerMixin 自带的 _build_train_dataloader 用
self.global_rank / self.world_size 做采样切分;Megatron 后端强制
TP=PP=CP=EP=1(见 dist.init_megatron_parallel_state),此时 DP rank ==
global rank,语义一致。后续放开模型并行时该 mixin 需改用 DP 维度。
"""

from deepspec.trainer.dspark_online_trainer import (
    DeepSeekV4FlashDSparkOnlineTrainer,
)
from deepspec.trainer.dspark_trainer import (
    DeepSeekV4FlashDSparkTrainer,
    Gemma4DSparkTrainer,
    Qwen3DSparkTrainer,
)
from deepspec.trainer.eagle3_trainer import (
    Gemma4Eagle3Trainer,
    Qwen3Eagle3Trainer,
)
from deepspec.trainer.megatron.base_trainer import MegatronBaseTrainer


class MegatronQwen3DSparkTrainer(Qwen3DSparkTrainer, MegatronBaseTrainer):
    """Qwen3 DSpark(离线 target cache)× Megatron 后端。"""


class MegatronGemma4DSparkTrainer(Gemma4DSparkTrainer, MegatronBaseTrainer):
    """Gemma4 DSpark(离线 target cache)× Megatron 后端。"""


class MegatronDeepSeekV4FlashDSparkTrainer(
    DeepSeekV4FlashDSparkTrainer, MegatronBaseTrainer
):
    """DeepSeek-V4-Flash DSpark(离线 target cache)× Megatron 后端。"""


class MegatronQwen3Eagle3Trainer(Qwen3Eagle3Trainer, MegatronBaseTrainer):
    """Qwen3 Eagle3 × Megatron 后端。"""


class MegatronGemma4Eagle3Trainer(Gemma4Eagle3Trainer, MegatronBaseTrainer):
    """Gemma4 Eagle3 × Megatron 后端。"""


class MegatronDeepSeekV4FlashDSparkOnlineTrainer(
    DeepSeekV4FlashDSparkOnlineTrainer, MegatronBaseTrainer
):
    """DeepSeek-V4-Flash 在线 DSpark(SGLang 取特征)× Megatron 后端。"""
