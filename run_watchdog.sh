#!/bin/bash
LOG=/tmp/watchdog.log
start_svc() {
  echo "[watchdog] $(date) restarting service" >> $LOG
  bash /sgl-workspace/sglang/scripts/killall_sglang.sh >/dev/null 2>&1
  pkill -9 -f sglang 2>/dev/null; sleep 5
  setsid bash /shenlb/zwf-spec/train-v1/deepspec/run_svc_v4flash.sh >/dev/null 2>&1 </dev/null &
  sleep 150
}
FAIL=0
echo "[watchdog] $(date) started" >> $LOG
while true; do
  code=$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then FAIL=0
  elif pgrep -f 'sglang.launch_server' >/dev/null 2>&1; then
    echo "[watchdog] $(date) health=$code proc-alive(warming)" >> $LOG
  else
    FAIL=$((FAIL+1)); echo "[watchdog] $(date) health=$code no-proc fail=$FAIL" >> $LOG
    [ $FAIL -ge 2 ] && { start_svc; FAIL=0; }
  fi
  sleep 30
done
