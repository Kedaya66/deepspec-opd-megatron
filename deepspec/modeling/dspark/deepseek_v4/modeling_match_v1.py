"""DeepSeek-V4 DSpark draft model, match-v1: forward math fully aligned with the
official ``inference/model.py`` (see layers_match_v1.py for the fix list).

Structural summary vs ``modeling.py``:
  * hidden state through the blocks is ``[B, S, hc_mult, D]`` (mHC streams);
    the noise embedding is expanded to hc_mult copies at the entrance, exactly
    like the reference ``forward_embed``;
  * a ``DSparkHyperHead`` collapses the streams before the final RMSNorm;
  * lm_head consumes the POST-norm hidden, while markov/confidence consume the
    PRE-norm (post-hyper-head) hidden — matching the reference ``forward_head``;
  * attention runs the inline eager path (sink), so ``_attn_implementation`` is
    forced to ``eager`` and the dense DSpark mask is always used.

Load official ``mtp.*`` weights with ``load_pretrained_match_v1.py`` (it also
applies the rope-channel permutation).

Eval note: ``draft_ops`` cache-based incremental decoding is NOT wired for this
variant yet (the match-v1 attention ignores ``past_key_values``); evaluate with
full-recompute forwards until that is adapted.
"""

from typing import Optional

import torch
from torch import nn

from deepspec.modeling.dspark.common import (
    AcceptRatePredictor,
    DSparkForwardOutput,
    build_eval_mask,
    create_dspark_attention_mask_dense,
    create_noise_embed,
    create_position_ids,
    log_sampler_stats,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.markov_head import build_markov_head
from deepspec.modeling.dspark.deepseek_v4.layers import (
    DeepSeekV4RMSNorm,
    DeepSeekV4RotaryEmbedding,
)
from deepspec.modeling.dspark.deepseek_v4.layers_match_v1 import (
    DeepSeekV4DSparkDecoderLayerMatchV1,
    DSparkHyperHead,
)
from deepspec.modeling.dspark.deepseek_v4.modeling import (
    DeepSeekV4DSparkModel,
    DeepSeekV4DSparkPreTrainedModel,
)


class DeepSeekV4DSparkModelMatchV1(DeepSeekV4DSparkModel):
    _no_split_modules = ["DeepSeekV4DSparkDecoderLayerMatchV1"]

    def __init__(self, config) -> None:
        # Bypass DeepSeekV4DSparkModel.__init__ (it builds the mismatched layer
        # stack); run the PreTrainedModel init directly, then build match-v1.
        config._attn_implementation = "eager"  # sink needs the inline eager path
        super(DeepSeekV4DSparkModel, self).__init__(config)
        self.config = config
        self.target_layer_ids = config.target_layer_ids
        self.hc_mult = int(getattr(config, "hc_mult", 4))

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [
                DeepSeekV4DSparkDecoderLayerMatchV1(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.hc_head = DSparkHyperHead(
            config.hidden_size,
            self.hc_mult,
            float(getattr(config, "hc_eps", 1e-6)),
            config.rms_norm_eps,
        )
        self.norm = DeepSeekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepSeekV4RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = DeepSeekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)

        self.markov_head = build_markov_head(config)

        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = False
        if self.enable_confidence_head:
            self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None
        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += int(config.markov_rank)
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)
        self.post_init()

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
        # Expand to hc_mult residual streams (reference forward_embed's repeat).
        hidden_states = noise_embedding.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        position_embeddings = self.rotary_emb(noise_embedding, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        collapsed = self.hc_head(hidden_states)      # [B, S, D], pre-norm
        normed = self.norm(collapsed)
        if return_prenorm:
            return normed, collapsed
        return normed

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> DSparkForwardOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(
            bsz, -1
        )
        draft_position_ids = create_position_ids(anchor_positions, self.block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        dspark_attn_mask = create_dspark_attention_mask_dense(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            device=device,
            sliding_window=getattr(self.config, "sliding_window", None),
        )
        output_hidden, prenorm_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
            return_prenorm=True,
        )

        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)
        prenorm_hidden_4d = prenorm_hidden.reshape(bsz, num_blocks, self.block_size, -1)

        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1,
                    anchor_positions.size(1),
                    -1,
                    -1,
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    target_last_hidden_states.size(-1),
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits = self.compute_logits(output_hidden).reshape(
            bsz,
            num_blocks,
            self.block_size,
            -1,
        )
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )

        log_sampler_stats(
            seq_len=seq_len,
            loss_mask=loss_mask,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
        )

        confidence_pred = None
        if self.confidence_head is not None:
            # Reference forward_head: confidence consumes the PRE-norm
            # (post-hyper-head) hidden.
            if self.confidence_head_with_markov:
                prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                    dtype=prenorm_hidden_4d.dtype
                )
                confidence_features = torch.cat(
                    [prenorm_hidden_4d, prev_embeddings],
                    dim=-1,
                )
                confidence_pred = self.confidence_head(confidence_features).float()
            else:
                confidence_pred = self.confidence_head(prenorm_hidden_4d).float()

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )


class DeepSeekV4ProDSparkModelMatchV1(DeepSeekV4DSparkModelMatchV1):
    """match-v1 draft for the DeepSeek-V4-Pro target."""


class DeepSeekV4FlashDSparkModelMatchV1(DeepSeekV4DSparkModelMatchV1):
    """match-v1 draft for the DeepSeek-V4-Flash target."""


__all__ = [
    "DeepSeekV4DSparkModelMatchV1",
    "DeepSeekV4ProDSparkModelMatchV1",
    "DeepSeekV4FlashDSparkModelMatchV1",
]
