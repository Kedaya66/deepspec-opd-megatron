#!/bin/bash
cd /shenlb/zwf-spec/train-v3/deepspec-opd
exec env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PORT=30000 TP=4 GPUS=0,1,2,3 MEM=0.85 CHUNK=102400 SWA_RATIO=0.3 \
  bash scripts/data/run_sglang_dspark_export.sh > /tmp/svc_opd.log 2>&1
