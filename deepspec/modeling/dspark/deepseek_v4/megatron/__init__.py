"""megatron-core 后端(TP / EP / fp8)。

**惰性导入**:megatron 只装在 pytorch-2606 容器里,而现有 FSDP2 训练路径跑在
sglang-dspark(torch 2.11,没有 megatron/TE)。这里如果 eager import megatron,
只要有人 import 到 deepspec.modeling.dspark.deepseek_v4 就会炸。
所以全部走 __getattr__ 惰性解析 —— 不碰 megatron 的代码路径不受影响。

convert_weights 不依赖 megatron(纯 torch),可以直接导。
"""

from deepspec.modeling.dspark.deepseek_v4.megatron.convert_weights import (
    convert_layer_state_dict,
    convert_model_state_dict,
)

_LAZY = {
    "DSparkParallelOptions": "mcore_config",
    "build_mcore_config": "mcore_config",
    "McoreDSparkAttention": "mcore_layers",
    "McoreDSparkLayer": "mcore_layers",
    "McoreDSparkHyperConnection": "mcore_layers",
    "McoreDSparkHyperHead": "mcore_layers",
    "init_dspark_parallel": "mcore_modeling",
    "build_dspark_layer_spec": "mcore_modeling",
    "DeepSeekV4DSparkModelMcore": "mcore_modeling",
    "DeepSeekV4FlashDSparkModelMcore": "mcore_modeling",
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    import importlib
    mod = importlib.import_module(
        f"deepspec.modeling.dspark.deepseek_v4.megatron.{_LAZY[name]}"
    )
    return getattr(mod, name)


__all__ = ["convert_layer_state_dict", "convert_model_state_dict", *_LAZY]
