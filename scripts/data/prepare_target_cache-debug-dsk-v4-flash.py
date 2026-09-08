import argparse
from dataclasses import dataclass
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer

from deepspec.data import ConversationCollator
from deepspec.data.target_cache_dataset import (
    AsyncTargetCacheWriter,
    LocalCacheWriteSummary,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    compute_local_sample_range,
    finalize_target_cache_index,
    load_local_cache_write_summary,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.utils import (
    CustomJSONEncoder,
    get_git_diff,
    get_git_sha,
    init_dist,
    is_global_main_process,
    load_config,
    main_process_first,
    parse_opts_to_config,
    print_on_global_main,
    print_on_local_main,
    seed_all,
)

os.environ["USE_TORCH"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PyTorch 2.10 Inductor still reads the legacy allow_tf32 flag while compiling.
torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class TargetForwardResult:
    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor


def _get_target_backbone(target_model):
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified"):
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model") and hasattr(target_model.model, "language_model"):
            return target_model.model.language_model
        assert False, "Gemma4 target model must expose a text language_model."
    return getattr(target_model, "model", target_model)


def _get_target_hidden_size(target_model) -> int:
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified"):
        return int(target_model.config.text_config.hidden_size)
    return int(target_model.config.hidden_size)


def _get_hook_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported target hook output type: {type(output)!r}")


def run_target_forward_with_hooks(
    *,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_ids,
):
    backbone = _get_target_backbone(target_model)
    layer_modules = backbone.layers
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    captured_hidden_states = {}
    handles = []

    def capture_layer(layer_id: int):
        def hook(_module, _inputs, output):
            captured_hidden_states[layer_id] = _get_hook_tensor(output).detach()

        return hook

    try:
        if -1 in target_layer_ids:
            handles.append(
                backbone.embed_tokens.register_forward_hook(capture_layer(-1))
            )
        for layer_id in target_layer_ids:
            if layer_id < 0:
                continue
            handles.append(
                layer_modules[layer_id].register_forward_hook(capture_layer(layer_id))
            )

        with torch.no_grad():
            target_output = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
            target_last_hidden_states = target_output.last_hidden_state.detach()
            # device_map 下各层 hidden state 可能分布在不同 GPU，cat 前先搬到同一设备
            _cat_device = target_last_hidden_states.device
            target_hidden_states = torch.cat(
                [
                    captured_hidden_states[layer_id].to(_cat_device)
                    for layer_id in target_layer_ids
                ],
                dim=-1,
            )
    finally:
        for handle in handles:
            handle.remove()
        captured_hidden_states.clear()

    return TargetForwardResult(
        target_hidden_states=target_hidden_states,
        target_last_hidden_states=target_last_hidden_states,
    )

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", required=True)
#     parser.add_argument("--opts", action="append", default=[])
#     parser.add_argument(
#         "--train-data-path",
#         action="append",
#         required=True,
#         help="Training JSONL path. Repeat this argument to use multiple files.",
#     )
#     parser.add_argument("--output-dir", required=True)
#     parser.add_argument("--min-loss-tokens", type=int, default=14)
#     parser.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3)
#     parser.add_argument("--local-batch-size", type=int, default=32)
#     parser.add_argument("--num-workers", type=int, default=4)
#     cli_args = parser.parse_args()
#     config = parse_opts_to_config(cli_args.opts, load_config(cli_args.config))
#     return cli_args, config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument(
        "--train-data-path",
        action="append",
        required=True,
        help="Training JSONL path. Repeat this argument to use multiple files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    # Debug run: hardcode the arguments (mirroring scripts/data/prepare_data.sh
    # Step 3/3) so the script can be launched directly under a debugger without
    # passing anything on the command line. Remove this list to fall back to the
    # real CLI (parser.parse_args()).
    debug_argv = [
        "--config", "/shenlb/zwf-spec/train-v1/deepspec/config/dspark/dspark_dskv4-flash.py",
        "--train-data-path",
        "/shenlb/zwf-spec/dataset/dataset-v1/主agent输入_医疗助手_正式环境_perfectblend_openai_conv_regen.jsonl",
        "--output-dir",
        os.path.expanduser("~/.cache/deepspec/DSK_V4_FLASH_target_cache"),
        "--local-batch-size", "16",
        # assistant 回复较短，默认 14 会丢弃约 4.66% 的超短样本；降到 4 保留大部分
        "--min-loss-tokens", "1",
    ]
    cli_args = parser.parse_args(debug_argv)
    config = parse_opts_to_config(cli_args.opts, load_config(cli_args.config))
    return cli_args, config


def _write_manifest(
    *,
    output_dir: str,
    config,
    train_data_paths,
    target_layer_ids,
    hidden_size: int,
    min_loss_tokens: int,
    shards,
):
    num_samples = sum(
        int(
            load_local_cache_write_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")
            )["num_local_samples"]
        )
        for rank in range(dist.get_world_size())
    )
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "target_model_name_or_path": str(config.model.target_model_name_or_path),
            "source_jsonl_paths": [str(path) for path in train_data_paths],
            "chat_template": str(config.data.chat_template),
            "max_length": int(config.data.max_length),
            "min_loss_tokens": int(min_loss_tokens),
            "project_name": (
                str(config.get("project_name"))
                if config.get("project_name") is not None
                else None
            ),
            "exp_name": (
                str(config.get("exp_name"))
                if config.get("exp_name") is not None
                else None
            ),
            "git_sha": str(get_git_sha()),
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)


def _print_prepare_progress(*, global_rank: int, processed_samples: int, total_samples: int):
    print(
        f"[prepare rank {global_rank}] {processed_samples}/{total_samples} samples",
        flush=True,
    )


def main(local_rank: int):
    cli_args, config = parse_args()
    train_data_paths = list(cli_args.train_data_path)
    target_layer_ids = [int(layer_id) for layer_id in config.model.target_layer_ids]
    min_loss_tokens = int(cli_args.min_loss_tokens)
    seed_all(int(config.seed))
    # 方案A：单进程 + device_map 模型并行。用 in-process 的 gloo 单 rank 组，
    # 让脚本里的 dist.barrier / broadcast / get_world_size 等调用仍可用。
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    global_rank, world_size = 0, 1
    device = None  # 模型 device_map 加载后再确定输入所在设备
    output_dir = os.path.abspath(cli_args.output_dir)
    print_on_local_main(json.dumps(config, indent=4, cls=CustomJSONEncoder), flush=True)
    print_on_local_main(
        json.dumps(
            {
                "train_data_path": train_data_paths,
                "output_dir": output_dir,
                "target_layer_ids": target_layer_ids,
                "min_loss_tokens": min_loss_tokens,
                "max_shard_bytes": int(cli_args.max_shard_bytes),
                "local_batch_size": int(cli_args.local_batch_size),
                "num_workers": int(cli_args.num_workers),
            },
            indent=4,
        ),
        flush=True,
    )
    if global_rank == 0:
        prepare_target_cache_output_dir(output_dir)
    dist.barrier()

    rank_dir = os.path.join(output_dir, "_tmp", f"rank_{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)

    with main_process_first():
        dataset = JsonLineDataset(data_paths=train_data_paths)

    local_start, local_end = compute_local_sample_range(
        num_samples=len(dataset),
        rank=global_rank,
        world_size=world_size,
    )
    local_total_samples = local_end - local_start

    local_subset = Subset(dataset, range(local_start, local_end))

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.target_model_name_or_path,
    )
    # DeepSeek-V4-Flash 不提供 jinja chat_template：接入官方 encode_messages（chat 模式）。
    # 官方编码在 chat 模式会生成 <｜Assistant｜></think>{content}<｜end▁of▁sentence｜>，
    # </think> 已由 v4-flash 模板的 assistant_loss_prefix 在 loss 中跳过。
    import sys as _sys
    _v4_enc_dir = os.path.join(str(config.model.target_model_name_or_path), "encoding")
    if _v4_enc_dir not in _sys.path:
        _sys.path.insert(0, _v4_enc_dir)
    from encoding_dsv4 import encode_messages as _v4_encode_messages
    from deepspec.data import parser as _deepspec_parser

    def _v4_render(_tokenizer, messages, *, add_generation_prompt, enable_thinking=None):
        def _flat(c):
            if isinstance(c, list):
                return "".join(b.get("text", "") for b in c if isinstance(b, dict))
            return c or ""
        msgs = [dict(m, content=_flat(m.get("content"))) for m in messages]
        return _v4_encode_messages(msgs, thinking_mode="chat", add_default_bos_token=True)

    _deepspec_parser.set_render_override(_v4_render)
    # target_model = AutoModel.from_pretrained(
    #     config.model.target_model_name_or_path,
    #     dtype=torch.bfloat16,
    #     attn_implementation="eager",
    #     device_map="auto",  # ← 方案A：按层自动切分到所有可见 GPU（模型并行）
    # ).eval()
    # # device_map 后模型跨多卡；输入送到 embedding 所在卡，accelerate 自动搬运跨卡激活
    # device = _get_target_backbone(target_model).embed_tokens.weight.device
    # target_hidden_size = _get_target_hidden_size(target_model)

    train_collator = ConversationCollator(
        tokenizer=tokenizer,
        chat_template=config.data.chat_template,
        max_length=config.data.max_length,
        min_loss_tokens=min_loss_tokens,
    )
    dataloader = DataLoader(
        local_subset,
        batch_size=int(cli_args.local_batch_size),
        collate_fn=train_collator,
        num_workers=int(cli_args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    writer = AsyncTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=int(cli_args.max_shard_bytes),
        max_queue_size=int(cli_args.local_batch_size) * 4,
    )

    processed_local_samples = 0
    last_progress_printed = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                processed_local_samples = min(
                    (batch_idx + 1) * int(cli_args.local_batch_size),
                    local_total_samples,
                )
                should_print_progress = (
                    processed_local_samples - last_progress_printed >= 100
                    or processed_local_samples == local_total_samples
                )
                if batch is None:
                    if should_print_progress:
                        _print_prepare_progress(
                            global_rank=global_rank,
                            processed_samples=processed_local_samples,
                            total_samples=local_total_samples,
                        )
                        last_progress_printed = processed_local_samples
                    continue
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }
                target_result = run_target_forward_with_hooks(
                    target_model=target_model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    target_layer_ids=target_layer_ids,
                )
                seq_lens = batch["attention_mask"].sum(dim=1).tolist()
                for sample_idx_in_batch, seq_len in enumerate(seq_lens):
                    seq_len = int(seq_len)
                    writer.write_sample(
                        input_ids=batch["input_ids"][sample_idx_in_batch, :seq_len],
                        attention_mask=batch["attention_mask"][
                            sample_idx_in_batch, :seq_len
                        ],
                        loss_mask=batch["loss_mask"][sample_idx_in_batch, :seq_len],
                        target_hidden_states=target_result.target_hidden_states[
                            sample_idx_in_batch, :seq_len
                        ],
                        target_last_hidden_states=target_result.target_last_hidden_states[
                            sample_idx_in_batch, :seq_len
                        ],
                    )
                if should_print_progress:
                    _print_prepare_progress(
                        global_rank=global_rank,
                        processed_samples=processed_local_samples,
                        total_samples=local_total_samples,
                    )
                    last_progress_printed = processed_local_samples
    finally:
        writer.close()
    del target_model
    torch.cuda.empty_cache()
    dataset.close()
    summary = LocalCacheWriteSummary(
        global_rank=global_rank,
        source_sample_start=local_start,
        source_sample_end=local_end,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))
    dist.barrier()

    shard_map = None
    summaries = None
    if is_global_main_process():
        summaries = [
            load_local_cache_write_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")
            )
            for rank in range(world_size)
        ]
        shard_map, shards = build_global_target_cache_shard_map(summaries)
    broadcast_payload = [shard_map]
    dist.broadcast_object_list(broadcast_payload, src=0)
    shard_map = broadcast_payload[0]
    local_summary = load_local_cache_write_summary(rank_dir)
    rename_local_target_cache_shards(
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=local_summary,
        shard_map=shard_map,
    )
    dist.barrier()

    if is_global_main_process():
        assert summaries is not None
        num_valid_samples = finalize_target_cache_index(
            output_dir=output_dir,
            summaries=summaries,
            shard_map=shard_map,
        )
        _write_manifest(
            output_dir=output_dir,
            config=config,
            train_data_paths=train_data_paths,
            target_layer_ids=target_layer_ids,
            hidden_size=target_hidden_size,
            min_loss_tokens=min_loss_tokens,
            shards=shards,
        )
        cleanup_target_cache_tmp_dir(output_dir)
        print_on_global_main(
            f"Prepared target cache at {output_dir} with "
            f"{num_valid_samples}/{len(dataset)} valid samples."
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # debug 模式：自动清空上次的输出目录，避免 FileExistsError
    import shutil
    _debug_cli_args, _ = parse_args()
    _debug_output_dir = os.path.abspath(_debug_cli_args.output_dir)
    if os.path.exists(_debug_output_dir):
        shutil.rmtree(_debug_output_dir)
        print(f"[debug] 已自动清空输出目录: {_debug_output_dir}")

    if os.path.exists(".git"):
        print("git status:", "\n\n".join(get_git_sha(detail_info=True)))
        print("git diff:", get_git_diff())
    # 方案A：单进程运行，模型通过 device_map 摊到全部 GPU（不再每卡一份副本）
    main(0)
