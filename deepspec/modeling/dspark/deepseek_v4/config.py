import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids


# DeepSeek-V4 MLA uses head_dim=512, which exceeds FlexAttention's triton
# shared-memory limit; use SDPA (with a dense DSpark mask) instead.
TRAIN_ATTN_IMPLEMENTATION = "sdpa"

# Fields the DeepSeek-V4 backbone reads from the (deep-copied) target config.
# They are asserted here so a mis-specified target config fails fast rather than
# deep inside the draft's first forward pass.
_REQUIRED_TARGET_FIELDS = (
    "hidden_size",
    "num_attention_heads",
    "num_hidden_layers",
    "head_dim",
    "qk_rope_head_dim",
    "q_lora_rank",
    "o_lora_rank",
    "o_groups",
    "moe_intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "scoring_func",
    "routed_scaling_factor",
    "rms_norm_eps",
    "rope_theta",
    "vocab_size",
)


def _build_draft_config(target_config, model_args, *, architecture: str):
    assert str(target_config.model_type) == "deepseek_v4", (
        "DeepSeek-V4 DSpark expects a deepseek_v4 target config, got "
        f"model_type={target_config.model_type!r}."
    )
    for field in _REQUIRED_TARGET_FIELDS:
        assert hasattr(target_config, field), (
            f"target_config.{field} must be provided for DeepSeek-V4 DSpark."
        )

    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    # The V4 draft attention ignores layer_types, but the DeepseekV4Config validator
    # (checked on save_pretrained) only allows its sparse-attention type names, so use
    # a valid one rather than "full_attention".
    layer_types = ["sliding_attention"] * num_draft_layers

    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )

    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = [architecture]
    draft_config.target_model_type = str(target_config.model_type)
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)

    # Truncate every per-layer config list (length == num_target_layers, e.g.
    # layer_types, mlp_layer_types) to the draft's layer count so DeepseekV4Config's
    # per-layer validators pass on save_pretrained. The V4 draft ignores these lists.
    for _attr, _val in list(vars(draft_config).items()):
        if isinstance(_val, list) and len(_val) == num_target_layers:
            setattr(draft_config, _attr, _val[:num_draft_layers])

    # The draft is dense bfloat16; drop the target's FP8 quantization metadata so a
    # saved checkpoint is not re-loaded through the FP8 quantizer on resume/eval.
    if getattr(draft_config, "quantization_config", None) is not None:
        # Remove the key entirely (setting it to None serializes as null, which
        # transformers.from_pretrained then chokes on).
        try:
            delattr(draft_config, "quantization_config")
        except AttributeError:
            draft_config.__dict__.pop("quantization_config", None)

    # The draft trains its MoE from scratch and has no token->expert hash table,
    # so hash routing (a serving feature of the target) is always disabled.
    draft_config.num_hash_layers = 0

    # Optional draft-side MoE overrides: the target model is very wide
    # (hundreds of experts); a draft can route far fewer without hurting the
    # speculative match rate. Absent overrides, the target's MoE shape is reused.
    if "draft_n_routed_experts" in model_args:
        draft_config.n_routed_experts = int(model_args.draft_n_routed_experts)
    if "draft_num_experts_per_tok" in model_args:
        draft_config.num_experts_per_tok = int(model_args.draft_num_experts_per_tok)
    if "draft_moe_intermediate_size" in model_args:
        draft_config.moe_intermediate_size = int(
            model_args.draft_moe_intermediate_size
        )
    assert int(draft_config.num_experts_per_tok) <= int(draft_config.n_routed_experts), (
        "num_experts_per_tok must not exceed n_routed_experts."
    )
    return draft_config


def build_pro_draft_config(target_config, model_args):
    return _build_draft_config(
        target_config,
        model_args,
        architecture="DeepSeekV4ProDSparkModel",
    )


def build_flash_draft_config(target_config, model_args):
    return _build_draft_config(
        target_config,
        model_args,
        architecture="DeepSeekV4FlashDSparkModel",
    )


# Default builder (Pro/Flash share identical config logic; the architecture name
# is the only difference).
build_draft_config = build_flash_draft_config


__all__ = [
    "build_draft_config",
    "build_pro_draft_config",
    "build_flash_draft_config",
]
