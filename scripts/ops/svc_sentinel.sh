#!/bin/bash
# 特征服务自愈哨兵(宿主机跑)。
# 魔改版 SGLang 在 hidden 导出长跑下会静默死锁(已复现 3 次,无异常栈)。
# 策略:每 60s 探测 /generate;连续 5 次失败 => 重启服务进程(只动进程,不动容器)。
# 训练侧配合:request_timeout=600 + 120 次重试 + NCCL 240min => 重启期间训练暂停而非崩死。
LOG=/tmp/svc_sentinel.log
fail=0
restarts=0
echo "[sentinel] start $(date -Is)" >> $LOG
while true; do
  code=$(timeout 30 curl -s -o /dev/null -m 25 -w "%{http_code}" -X POST http://127.0.0.1:30000/generate \
    -H 'Content-Type: application/json' \
    -d '{"input_ids":[1,2,3],"sampling_params":{"max_new_tokens":2,"temperature":0.0}}' 2>/dev/null)
  if [ "$code" = "200" ]; then
    fail=0
  else
    # 2026-08-13:满负荷时探测会排队超时(假阴性)。死锁的鉴别特征是
    # scheduler 日志停止滚动 —— 60s 内有新写入就算活着(忙 != 死)。
    slog=$(docker exec sglang-dspark bash -c 'ls -t /tmp/svc_tp4_0731*.log 2>/dev/null | head -1')
    # 2026-08-13 v2:死锁的服务仍会刷 state-deleted 余波,单看"日志在滚"会误判为忙。
    # 有效活性 = 最近日志里有真实的 batch 处理行。
    fresh=$(docker exec sglang-dspark bash -c "tail -c 200000 $slog 2>/dev/null | tail -30 | grep -ac 'Prefill batch\|Decode batch'")
    if [ "${fresh:-0}" -ge 1 ]; then
      fail=0
      echo "[sentinel] probe $code 但服务日志在滚(忙),不计失败 $(date -Is)" >> $LOG
    else
      fail=$((fail+1))
      echo "[sentinel] probe fail #$fail (code=$code, 日志停滚) $(date -Is)" >> $LOG
    fi
  fi
  if [ $fail -ge 5 ]; then
    restarts=$((restarts+1))
    echo "[sentinel] RESTART #$restarts begin $(date -Is)" >> $LOG
    docker exec sglang-dspark bash -c 'P=$(pgrep -f "^python3 -m sglang.launch_server" | head -1); [ -n "$P" ] && kill $P; sleep 8; P2=$(pgrep -f "^python3 -m sglang.launch_server"); S=$(pgrep -f "sglang::scheduler"); [ -n "$P2" ] && kill -9 $P2 2>/dev/null; [ -n "$S" ] && kill -9 $S 2>/dev/null; sleep 5' >> $LOG 2>&1  # 2026-08-13: 死锁服务连 SIGTERM 都挂起,须补 kill -9
    # 等 GPU0-3 排空
    for i in $(seq 1 30); do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 | sort -n | tail -1)
      [ "${u:-999999}" -lt 4000 ] && break
      sleep 10
    done
    docker exec sglang-dspark bash -c 'cd /shenlb/zwf-spec/train-v1/deepspec && MODEL=/public_andon-AI/deepseek-ai/DeepSeek-V4-Flash-0731 PORT=30000 TP=4 GPUS=0,1,2,3 MEM=0.85 CHUNK=102400 SWA_RATIO=0.3 setsid nohup bash scripts/data/run_sglang_dspark_export.sh > /tmp/svc_tp4_0731_s'$restarts'.log 2>&1 < /dev/null &' >> $LOG 2>&1
    # 等就绪(JIT 后首 200,最多 40min)
    for i in $(seq 1 120); do
      code=$(timeout 20 curl -s -o /dev/null -m 15 -w "%{http_code}" -X POST http://127.0.0.1:30000/generate \
        -H 'Content-Type: application/json' \
        -d '{"input_ids":[1,2,3],"sampling_params":{"max_new_tokens":2,"temperature":0.0}}' 2>/dev/null)
      [ "$code" = "200" ] && break
      sleep 20
    done
    echo "[sentinel] RESTART #$restarts done code=$code $(date -Is)" >> $LOG
    fail=0
  fi
  sleep 60
done
