#!/bin/bash
# [10/16] ABot smoke on the shared GPU 6 under the lock (PROTOCOL Section 5): one of ABot's own images x P2 (turn) x seed 1234,
# run A = official low-memory path (T5 DynamicSwap offload, --offload-t5 on), run B = T5 resident (only if >= 35 GB are free at lock time).
# Each run takes the lock separately and finishes in ~3 min. Quant = official default fp8-per-token; falls back to fp8-per-tensor if it errors.
# usage: nohup /home/hk28965/specwm/baselines/abot/smoke_gpu6.sh > /home/hk28965/specwm/baselines/abot/smoke_gpu6.log 2>&1 &
S=/home/hk28965/specwm; B=$S/baselines; IMG=$B/scenes/abot/desert_valley_3rd.png; SC=abot__desert_valley_3rd__P2; Q=fp8-per-token
run() {  # $1 = out dir, $2 = offload mode, $3 = quant
  local t0=$(date +%s); echo "[$(date '+%H:%M:%S')] waiting for /tmp/gpu6.lock: $1 offload=$2 quant=$3"
  flock /tmp/gpu6.lock bash -c "echo \"[\$(date '+%H:%M:%S')] lock acquired; GPU6 \$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader -i 6)\"; \
    if [ '$2' = off ] && [ \$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 6) -lt 35000 ]; then echo SKIP_NOT_ENOUGH_FREE_VRAM; exit 3; fi; \
    t1=\$(date +%s); CUDA_VISIBLE_DEVICES=6 $B/abot/run.sh $IMG P2 1234 $1 --offload-t5 $2 --quant $3; rc=\$?; echo \"[\$(date '+%H:%M:%S')] run exit \$rc, wall under lock \$(( \$(date +%s) - t1 )) s\"; exit \$rc"
  return $?
}
mkdir -p $B/runs/abot_t5offload $B/runs/abot
run $B/runs/abot_t5offload/$SC on $Q; rc=$?
if [ $rc -ne 0 ] || [ ! -f $B/runs/abot_t5offload/$SC/video.mp4 ]; then echo "run A failed with $Q (rc $rc) -> retry with fp8-per-tensor"; Q=fp8-per-tensor; mv $B/runs/abot_t5offload/$SC $B/runs/abot_t5offload/${SC}_failed_pertoken 2>/dev/null; run $B/runs/abot_t5offload/$SC on $Q; fi
run $B/runs/abot/$SC off $Q
echo "[$(date '+%H:%M:%S')] SMOKE_DONE quant=$Q"
