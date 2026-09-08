from .base_trainer import BaseTrainer
from .dspark_trainer import (
    DeepSeekV4FlashDSparkTrainer,
    Gemma4DSparkTrainer,
    Qwen3DSparkTrainer,
)
from .dspark_online_trainer import DeepSeekV4FlashDSparkOnlineTrainer
from .eagle3_trainer import Gemma4Eagle3Trainer, Qwen3Eagle3Trainer

_MEGATRON_TRAINERS = (
    "MegatronBaseTrainer",
    "MegatronDeepSeekV4FlashDSparkOnlineTrainer",
    "MegatronDeepSeekV4FlashDSparkTrainer",
    "MegatronGemma4DSparkTrainer",
    "MegatronGemma4Eagle3Trainer",
    "MegatronQwen3DSparkTrainer",
    "MegatronQwen3Eagle3Trainer",
)

__all__ = [
    "BaseTrainer",
    "Gemma4Eagle3Trainer",
    "Gemma4DSparkTrainer",
    "Qwen3Eagle3Trainer",
    "Qwen3DSparkTrainer",
    "DeepSeekV4FlashDSparkTrainer",
    "DeepSeekV4FlashDSparkOnlineTrainer",
    *_MEGATRON_TRAINERS,
]


def __getattr__(name):
    # Megatron 系列 trainer 惰性导入:不装 megatron-core 也能用 FSDP 路径
    if name in _MEGATRON_TRAINERS:
        import deepspec.trainer.megatron as _megatron

        return getattr(_megatron, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
