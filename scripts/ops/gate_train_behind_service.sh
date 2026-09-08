#!/bin/bash
# 闸门:等服务真正可服务(连续 3 次 200)才放训练流量,然后启动 512 系列。
ok=0
for i in $(seq 1 150); do
  code=$(timeout 20 curl -s -o /dev/null -m 15 -w "%{http_code}" -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" -d "{\"input_ids\":[1,2,3],\"sampling_params\":{\"max_new_tokens\":2,\"temperature\":0.0}}" 2>/dev/null)
  if [ "$code" = "200" ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 3 ] && { echo "[gate] 服务连续 3 次 200,放行 $(date -Is)"; exec /root/drive_512.sh; }
  sleep 20
done
echo "[gate] 50 分钟未就绪,放弃"
