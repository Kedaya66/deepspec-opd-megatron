#!/bin/bash
cd /shenlb/zwf-spec/train-v3/deepspec-opd
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_TORCH=true WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python train.py --config config/dspark/dspark_dskv4-flash_opd.py 2>&1 | tee /tmp/train_opd.log
