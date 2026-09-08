from deepspec.data import CacheCollator
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.trainer.base_trainer import BaseTrainer
from deepspec.modeling.dspark.deepseek_v4 import DeepSeekV4FlashDSparkModel
from deepspec.modeling.dspark.deepseek_v4.config import build_flash_draft_config
from deepspec.modeling.dspark.deepseek_v4.load_pretrained import (
    apply_pretrained_dspark_weights,
)
from deepspec.utils import print_on_global_main
import torch
from transformers import AutoConfig, AutoTokenizer
from deepspec.modeling.dspark.deepseek_v4.load_pretrained import load_embed_and_head


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )
        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)


def _maybe_load_pretrained_dspark(model, model_args):
    """可选：从已发布的 DSpark checkpoint 热启动 draft 权重。

    由 model.pretrained_dspark_path 控制(缺省/None -> 随机初始化);
    load_pretrained_weights 门控(给了路径时默认 True)。在 .to(device)/FSDP 之前于 CPU 加载。
    """
    path = (
        model_args.pretrained_dspark_path
        if "pretrained_dspark_path" in model_args
        else None
    )
    if not path:
        return model
    enabled = (
        bool(model_args.load_pretrained_weights)
        if "load_pretrained_weights" in model_args
        else True
    )
    if not enabled:
        print_on_global_main(
            f"[dspark] pretrained_dspark_path set ({path}) but "
            "load_pretrained_weights=False; keeping random init."
        )
        return model
    print_on_global_main(
        f"[dspark] warm-starting draft from pretrained checkpoint: {path}"
    )
    backend = str(model_args.backend if "backend" in model_args else "").lower()
    if backend == "mcore":
        # mcore 模型的参数布局和 HF 不同(专家 gate/up 拼成 linear_fc1、wo_a 变三维),
        # 而且专家要按 ep_rank 只取本 rank 那份。所以先拿 HF layout 的 state_dict,
        # 过 convert_model_state_dict 再灌。
        # 转换本身已做过数值对拍(单层输出最大相对差 1.8e-4)和 EP 切片检查。
        import torch.distributed as dist
        from megatron.core import parallel_state

        from deepspec.modeling.dspark.deepseek_v4.load_pretrained_match_v1 import (
            build_match_v1_state_dict,
        )
        from deepspec.modeling.dspark.deepseek_v4.megatron import (
            convert_model_state_dict,
        )

        hf_sd = build_match_v1_state_dict(str(path), device="cpu")
        mc = dict(model_args.mcore) if "mcore" in model_args else {}
        cfg = model.config
        conv = convert_model_state_dict(
            hf_sd,
            num_layers=int(cfg.num_hidden_layers),
            num_heads=int(cfg.num_attention_heads),
            head_dim=int(cfg.head_dim),
            n_groups=int(cfg.o_groups),
            o_lora_rank=int(cfg.o_lora_rank),
            num_experts=int(cfg.n_routed_experts),
            tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            tp_size=parallel_state.get_tensor_model_parallel_world_size(),
            ep_rank=parallel_state.get_expert_model_parallel_rank(),
            ep_size=parallel_state.get_expert_model_parallel_world_size(),
            moe_grouped_gemm=bool(mc.get("moe_grouped_gemm", True)),
        )
        res = model.load_state_dict(conv, strict=False)
        missing = [k for k in res.missing_keys if "_extra_state" not in k]
        # 漏权重是静默的(strict=False 下保持随机初始化),必须显式拦。
        # 但要区分两种 missing:
        #   * layers.* —— 经过转换的,漏了就是转换 bug,必须拦
        #   * 其余 —— 预训练 ckpt 本来就没有的头(confidence_head 是本项目
        #     从零训的,4353 个参数,官方 DSpark ckpt 里没有),HF 路径同样
        #     靠 strict=False 跳过并保持随机初始化,属正常
        dropped = [k for k in missing if k.startswith("layers.")]
        assert not dropped, (
            f"转换后有 {len(dropped)} 个 decoder 层参数没被覆盖 —— 这是转换 bug"
            f"(前 10): {dropped[:10]}"
        )
        fresh = [k for k in missing if not k.startswith("layers.")]
        unexpected_in_ckpt = [k for k in fresh if k in hf_sd]
        assert not unexpected_in_ckpt, (
            f"这些参数预训练里有却没进模型,说明顶层搬运丢了键: {unexpected_in_ckpt[:10]}"
        )
        print_on_global_main(
            f"[dspark] mcore 权重已转换加载:{len(conv)} 个张量,"
            f"ep_rank={parallel_state.get_expert_model_parallel_rank()}"
            f"/{parallel_state.get_expert_model_parallel_world_size()},"
            f"unexpected={len(res.unexpected_keys)}"
            + (f",随机初始化(预训练里没有): {fresh}" if fresh else "")
        )
    elif bool(getattr(model_args, "match_v1", True)):
        # match-v1 加载器:比 base 多保留 attn_sink/hc_* 权重,并做一次 rope 通道
        # 置换(只针对官方 layout;match-v1 存出的 ckpt 走 resume 路径,不再过这里)
        from deepspec.modeling.dspark.deepseek_v4.load_pretrained_match_v1 import (
            apply_pretrained_match_v1_weights,
        )

        apply_pretrained_match_v1_weights(model, str(path), device="cpu")
    else:
        apply_pretrained_dspark_weights(model, str(path), device="cpu")
    return model


class DeepSeekV4FlashDSparkTrainer(Qwen3DSparkTrainer):
    """DeepSeek-V4-Flash 的 DSpark draft trainer。

    复用 Qwen3DSparkTrainer 的训练循环/run_batch(forward 签名一致),
    仅换 draft 模型为 DeepSeekV4FlashDSparkModel,并支持加载预训练 DSpark 权重。
    走离线 CacheCollator(与父类相同)。
    """

    def build_models(self):
        """Init & freeze embed/lm_head from the bf16 DSpark checkpoint, skipping the
        148GB fp8 target CPU load in base (the checkpoint already ships embed/head)."""
        model_args = self.args.model
        tokenizer = AutoTokenizer.from_pretrained(model_args.target_model_name_or_path)
        target_config = AutoConfig.from_pretrained(model_args.target_model_name_or_path)
        draft_model = self._build_draft_model(
            target_config=target_config, model_args=model_args
        )
        if bool(getattr(model_args, "gradient_checkpointing", False)):
            try:
                draft_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                print_on_global_main("[dspark] gradient checkpointing enabled")
            except Exception as exc:  # noqa: BLE001
                print_on_global_main(
                    f"[dspark] gradient checkpointing NOT enabled: {exc}"
                )
        src = (
            model_args.pretrained_dspark_path
            if "pretrained_dspark_path" in model_args
            else None
        )
        assert src, (
            "DeepSeekV4FlashDSparkTrainer needs model.pretrained_dspark_path (bf16 "
            "checkpoint) to initialize embed/lm_head."
        )
        embed_w, head_w = load_embed_and_head(str(src), device="cpu")
        with torch.no_grad():
            draft_model.embed_tokens.weight.copy_(
                embed_w.to(draft_model.embed_tokens.weight.dtype)
            )
            draft_model.lm_head.weight.copy_(
                head_w.to(draft_model.lm_head.weight.dtype)
            )
        draft_model.set_embedding_head_trainable(False)
        print_on_global_main(
            f"[dspark] embed/lm_head initialized from bf16 checkpoint (frozen): {src}"
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_flash_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        backend = str(
            model_args.backend if "backend" in model_args else ""
        ).lower()
        if backend == "mcore":
            # megatron-core 后端:只换 decoder 层(内含 mcore MoELayer,带 EP),
            # embedding/heads/loss 全部继承 match-v1。模型构造时会调
            # init_dspark_parallel 建并行组 —— 它对已初始化的情况直接 return,
            # 所以 train.megatron 那边的并行度必须与 model.mcore 一致,
            # 否则先建组的那份说了算(见 megatron/dist.py 里的一致性断言)。
            from deepspec.modeling.dspark.deepseek_v4.megatron import (
                DSparkParallelOptions,
                DeepSeekV4FlashDSparkModelMcore,
            )

            mcore_cfg = dict(model_args.mcore) if "mcore" in model_args else {}
            opts = DSparkParallelOptions(**mcore_cfg)
            draft_config.architectures = ["DeepSeekV4FlashDSparkModelMcore"]
            model = DeepSeekV4FlashDSparkModelMcore(
                draft_config, parallel_opts=opts
            )
        elif bool(getattr(model_args, "match_v1", True)):
            # match-v1:前向数学与官方 inference/model.py 对齐(mHC 残差流/
            # attention sink/V-RoPE)。eval 侧尚未接增量解码,只能全量前向。
            # 本仓默认 True;显式 model.match_v1=false 才回退 base 实现。
            from deepspec.modeling.dspark.deepseek_v4.modeling_match_v1 import (
                DeepSeekV4FlashDSparkModelMatchV1,
            )

            draft_config.architectures = ["DeepSeekV4FlashDSparkModelMatchV1"]
            model = DeepSeekV4FlashDSparkModelMatchV1(draft_config)
        else:
            model = DeepSeekV4FlashDSparkModel(draft_config)
        return _maybe_load_pretrained_dspark(model, model_args)
