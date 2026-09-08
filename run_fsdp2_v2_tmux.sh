#!/usr/bin/env bash
set -euo pipefail

cd /shenlb/zwf-spec/train-v1/deepspec-v1-fsdp2-v2
mkdir -p .run_logs

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MASTER_PORT=29983
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_TORCH=true
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONFAULTHANDLER=1

exec > >(tee -a .run_logs/fsdp2_v2_train.log) 2>&1
echo "[launcher] start $(date -Is)"
python3 train.py --config config/dspark/dspark_dskv4-flash.py
status=$?
echo "[launcher] exit=$status $(date -Is)"
exit "$status"
