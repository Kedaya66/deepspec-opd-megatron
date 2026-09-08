import os

## cache dir
CACHE_DIR = os.path.expanduser("~/.cache/deepspec")

## model_name_or_path
# DSK_V4_FLASH = "/andon-AI/ai-infra-benchmark/model/DeepSeek-V4-Flash-DSpark"
# 2026-08-12 换 target 权重到 0731(旧目录已在 cubefs 故障期间丢失)
# DSK_V4_FLASH = "/public_andon-AI/deepseek-ai/DeepSeek-V4-Flash"
DSK_V4_FLASH = "/public_andon-AI/deepseek-ai/DeepSeek-V4-Flash-0731"
QWEN_3_4B = "/andon_AI/Qwen/Qwen3-4B"
QWEN_3_8B = "/andon_AI/Qwen/Qwen3-8B"
QWEN_3_14B = "/andon_AI/Qwen/Qwen3-14B"
GEMMA_4_12B = "google/gemma-4-12B-it"
BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")

## auto eval
auto_eval_command = None