#!/bin/bash
# 解耦部署：独立起 SGLang 服务用于 target 特征提取（dspark hidden-state export）。
# 与训练/取特征脚本解耦——服务常驻，客户端 POST /generate 取特征。
#
# 用法：
#   ./run_sglang_dspark_export.sh            # tp8，占满 8 卡，端口 30000
#   TP=2 GPUS=0,1 PORT=30000 ./run_sglang_dspark_export.sh   # tp2 单实例（2 卡），做数据并行时起多个
set -euo pipefail

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 关键：开启 dspark 三层导出（逐 prompt token 返回 cat([L40,L41,L42,last_norm])）
export SGLANG_DSPARK_EXPORT_LAYERS="${SGLANG_DSPARK_EXPORT_LAYERS:-40,41,42}"

MODEL="${MODEL:-/public_andon-AI/deepseek-ai/DeepSeek-V4-Flash}"
TP="${TP:-8}"
PORT="${PORT:-30000}"
MEM="${MEM:-0.85}"
GPUS="${GPUS:-}"   # 如 "0,1"；多实例数据并行时用它把每个实例绑到不同卡

[ -n "$GPUS" ] && export CUDA_VISIBLE_DEVICES="$GPUS"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --tp "$TP" \
  --moe-runner-backend marlin \
  --enable-return-hidden-states \
  --disable-cuda-graph \
  --disable-radix-cache \
  --chunked-prefill-size ${CHUNK:-102400} \
  --mem-fraction-static "$MEM" \
  --swa-full-tokens-ratio "${SWA_RATIO:-0.1}" \
  --watchdog-timeout "${WATCHDOG:-300}" \
  --skip-server-warmup \
  --host 0.0.0.0 --port "$PORT"
