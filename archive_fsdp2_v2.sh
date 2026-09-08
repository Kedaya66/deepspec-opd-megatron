#!/usr/bin/env bash
set -u

SRC=/root/checkpoints/deepspec/dspark_v1_matchv1_bf16_fsdp2_v2
DST=/public_andon-AI/ai-infra-benchmark/trained/dspark_v1_matchv1_bf16_fsdp2_v2
LOG=/shenlb/zwf-spec/train-v1/deepspec-v1-fsdp2-v2/.run_logs/fsdp2_v2_archive.log
TARGETS="5 6 11 12 13 14 15 16 17 18 19 20 21 22"
declare -A last_size

mkdir -p "$DST" "$(dirname "$LOG")"
echo "[archive] $(date -Is) start targets=$TARGETS" >> "$LOG"

while true; do
  done_all=1
  for step in $TARGETS; do
    src_dir="$SRC/step_$step"
    dst_dir="$DST/step_$step"
    src_model="$src_dir/model.safetensors"
    dst_model="$dst_dir/model.safetensors"

    if [ -f "$dst_model" ]; then
      continue
    fi
    done_all=0
    if [ ! -f "$src_model" ]; then
      continue
    fi

    size=$(stat -c %s "$src_model" 2>/dev/null || echo 0)
    if [ "$size" -lt 38000000000 ] || [ "${last_size[$step]:-}" != "$size" ]; then
      last_size[$step]=$size
      continue
    fi

    tmp_dir="$DST/.step_${step}.tmp.$$"
    rm -rf "$tmp_dir"
    mkdir -p "$tmp_dir"
    if cp "$src_model" "$src_dir/config.json" "$src_dir/train_config.py" "$tmp_dir/" 2>> "$LOG"; then
      copied_size=$(stat -c %s "$tmp_dir/model.safetensors" 2>/dev/null || echo 0)
      if [ "$copied_size" = "$size" ]; then
        mv "$tmp_dir" "$dst_dir"
        echo "[archive] $(date -Is) OK step_$step ($((size / 1000000000))G)" >> "$LOG"
        continue
      fi
    fi
    rm -rf "$tmp_dir"
    echo "[archive] $(date -Is) RETRY step_$step" >> "$LOG"
  done

  if [ "$done_all" = 1 ]; then
    echo "[archive] $(date -Is) all done" >> "$LOG"
    exit 0
  fi
  sleep 60
done
