import argparse
import json
import os
import torch
from deepspec.utils import (
    CustomJSONEncoder,
    get_git_diff,
    load_config,
    parse_opts_to_config,
    seed_all,
    get_git_sha,
)

os.environ['USE_TORCH']='true'
os.environ['WANDB_DISABLED']='true'
os.environ['TOKENIZERS_PARALLELISM']='false'
torch.set_float32_matmul_precision("high")


_TARGET_CACHE_DIR = "/shenlb/zwf-spec/train-v1/deepspec/target_cache/qwen3_4b"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--opts", action="append", default=[])

    # debug 模式：直接硬编码参数，无需命令行传参
    import sys
    if len(sys.argv) == 1:
        debug_argv = [
            "--config", "/shenlb/zwf-spec/train-v1/deepspec/config/dspark/dspark_qwen3_4b.py",
            "--opts", f"data.target_cache_path={_TARGET_CACHE_DIR}",
            "--opts", "train.global_batch_size=4",
            "--opts", "train.local_batch_size=1",
        ]
        args = parser.parse_args(debug_argv)
    else:
        args = parser.parse_args()

    config = parse_opts_to_config(args.opts, load_config(args.config))
    config._origin_config_path = os.path.abspath(args.config)
    config._origin_opts = list(args.opts)
    return config


def main(local_rank):
    args = parse_args()
    seed_all(int(args.seed))
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    trainer = args.train.trainer_cls(local_rank, args)
    trainer.train()
    trainer.clean_up()


if __name__ == "__main__":
    if os.path.exists(".git"):
        print("git status:", "\n\n".join(get_git_sha(detail_info=True)))
        print("git diff:", get_git_diff())
    # debug 模式只用 0~3 卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    torch.multiprocessing.spawn(main, nprocs=4)
