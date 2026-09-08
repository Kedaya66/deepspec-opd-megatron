#!/bin/bash
cd /shenlb/zwf-spec/train-v1/deepspec
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_TORCH=true WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
echo "[launch] $(date) starting full train" > /tmp/full_train.log
exec /usr/bin/python train.py --config config/dspark/dspark_dskv4-flash.py >> /tmp/full_train.log 2>&1
