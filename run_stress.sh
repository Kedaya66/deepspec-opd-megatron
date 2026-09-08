#!/bin/bash
cd /shenlb/zwf-spec/train-v1/deepspec
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_TORCH=true WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
echo "[stress] $(date) start" > /tmp/stress.log
exec /usr/bin/python train.py --config config/dspark/dspark_stress_offload.py >> /tmp/stress.log 2>&1
