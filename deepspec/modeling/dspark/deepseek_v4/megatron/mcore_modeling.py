"""megatron-core 版 DSpark draft 模型。

移植边界:**只换 decoder 层**。embedding、fc/hidden_norm、final norm、lm_head、
markov head、confidence head、anchor 采样、mask 构造、loss —— 全部继承 HF 版不动。
理由:那些跟并行无关,换掉没收益只有风险;而 decoder 层里的 MoE 才是 EP/fp8 的战场。

所以本文件只干三件事:
  1) 把 self.layers 换成 McoreDSparkLayer(内含 mcore MoELayer)
  2) 把 self.hc_head 换成 McoreDSparkHyperHead(纯搬家,数值一致)
  3) 重写 _forward_backbone —— mcore 层的签名和返回值和 HF 层不同
"""

from typing import Optional

import torch
from torch import nn

from megatron.core import parallel_state
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from deepspec.modeling.dspark.deepseek_v4.modeling_match_v1 import (
    DeepSeekV4DSparkModelMatchV1,
)
from deepspec.modeling.dspark.deepseek_v4.megatron.mcore_config import (
    DSparkParallelOptions,
    build_mcore_config,
)
from deepspec.modeling.dspark.deepseek_v4.megatron.mcore_layers import (
    McoreDSparkHyperHead,
    McoreDSparkLayer,
)


def init_dspark_parallel(opts: DSparkParallelOptions) -> None:
    """建模型之前必须先建并行组 —— ColumnParallelLinear.__init__ 就会去问它。"""
    if not torch.distributed.is_initialized():
        raise RuntimeError("先 torch.distributed.init_process_group,再调这个。")
    if parallel_state.model_parallel_is_initialized():
        return
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=opts.tensor_model_parallel_size,
        pipeline_model_parallel_size=1,          # 3 层 draft,PP 没意义
        expert_model_parallel_size=opts.expert_model_parallel_size,
        expert_tensor_parallel_size=opts.expert_tensor_parallel_size,
        # 2026-08-12:必须显式拉长。此处先建组则 trainer dist.py 的 60min 被跳过
        # ("先建组的一方说了算"),而 megatron 默认 10min —— OPD 数据管线等
        # rollout 超 10min 时,先进 all_to_all 的 rank 会被 NCCL watchdog 带崩
        # (0731 权重生成变长后 gbs=512 实测复现,12.3min 处炸 EP 组)。
        # 2026-08-13 60->240:特征服务会间歇死锁(自愈哨兵约 5-8min 重启),
        # 数据等待是合法状态,NCCL 不应把"等数据"判成通信失败
        distributed_timeout_minutes=240,
    )


def build_dspark_layer_spec(mcore_config, opts: DSparkParallelOptions) -> ModuleSpec:
    """层的零件清单。只填 mlp 一格 —— 就为了拿 mcore 的 MoELayer(router+EP 全在里面)。"""
    if opts.use_transformer_engine:
        # TESpecProvider 不在 models.backends 里(那儿只有 Local / Inference),
        # 它住在 extensions 下 —— 因为它依赖 transformer_engine 这个外部包。
        from megatron.core.extensions.transformer_engine_spec_provider import (
            TESpecProvider,
        )
        backend = TESpecProvider()
    else:
        backend = LocalSpecProvider()

    if mcore_config.num_moe_experts:
        mlp = get_moe_module_spec_for_backend(
            backend=backend,
            num_experts=mcore_config.num_moe_experts,
            moe_grouped_gemm=mcore_config.moe_grouped_gemm,
        )
    else:
        mlp = get_mlp_module_spec_for_backend(backend=backend, num_experts=None)
    return ModuleSpec(
        module=McoreDSparkLayer,
        submodules=TransformerLayerSubmodules(mlp=mlp),
    )


class DeepSeekV4DSparkModelMcore(DeepSeekV4DSparkModelMatchV1):
    """match-v1 语义 + megatron-core 后端(TP / EP / fp8)。"""

    _no_split_modules = ["McoreDSparkLayer"]

    def __init__(self, config, parallel_opts: Optional[DSparkParallelOptions] = None):
        # 先按 HF 版建全套(embedding/heads/loss 都要),再把 decoder 换掉。
        # 多建一次 HF 层再丢弃会白占内存,所以先把层数改成 0 骗过父类。
        num_layers = int(config.num_hidden_layers)
        config.num_hidden_layers = 0
        super().__init__(config)
        config.num_hidden_layers = num_layers

        self.parallel_opts = parallel_opts or DSparkParallelOptions()
        self.mcore_config = build_mcore_config(config, self.parallel_opts)
        init_dspark_parallel(self.parallel_opts)

        pgs = ProcessGroupCollection.use_mpu_process_groups()
        spec = build_dspark_layer_spec(self.mcore_config, self.parallel_opts)
        self.layers = nn.ModuleList([
            build_module(spec, config=self.mcore_config,
                         layer_number=i + 1, pg_collection=pgs)
            for i in range(num_layers)
        ])
        # hc_head 换成同名 mcore 版(结构、参数形状、数值完全一致,纯搬家)
        self.hc_head = McoreDSparkHyperHead(
            config.hidden_size,
            self.hc_mult,
            float(getattr(config, "hc_eps", 1e-6)),
            config.rms_norm_eps,
        )
        self.config.num_hidden_layers = num_layers

    # HF 层签名是 (hidden_states, target_hidden_states, attention_mask,
    # position_ids, position_embeddings) 且返回单张量;mcore 层是
    # (hidden_states, target_hidden_states, cos, sin, attention_mask) 返回二元组。
    # 只有这一处需要重写。
    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden_states: Optional[torch.Tensor] = None,
        return_prenorm: bool = False,
        **kwargs,
    ):
        hidden_states = noise_embedding.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        cos, sin = self.rotary_emb(noise_embedding, position_ids)
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                target_hidden_states=target_hidden_states,
                cos=cos, sin=sin,
                attention_mask=attention_mask,
            )
        collapsed = self.hc_head(hidden_states)
        normed = self.norm(collapsed)
        if return_prenorm:
            return normed, collapsed
        return normed


class DeepSeekV4FlashDSparkModelMcore(DeepSeekV4DSparkModelMcore):
    pass


__all__ = [
    "init_dspark_parallel",
    "build_dspark_layer_spec",
    "DeepSeekV4DSparkModelMcore",
    "DeepSeekV4FlashDSparkModelMcore",
]
