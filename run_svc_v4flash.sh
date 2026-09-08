#!/bin/bash
cd /shenlb/zwf-spec/train-v1/deepspec
exec env PORT=30000 TP=4 GPUS=0,1,2,3 bash scripts/data/run_sglang_dspark_export.sh > /tmp/svc_tp4.log 2>&1
