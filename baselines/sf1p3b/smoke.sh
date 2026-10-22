#!/bin/bash
# [baselines/sf1p3b] one-off smoke on the shared GPU 6 (PROTOCOL §5): tf (with the untimed oracle calibration) then uniform:4 on the same clip.
# Runs under flock /tmp/gpu6.lock; every generation process has its own hard timeout (TIMEOUT -> `timeout` on the python process in run.sh) so the
# lock is held < 15 min in total. usage: smoke.sh OUTDIR [IMAGE] [PROGRAM]
S=/home/hk28965/specwm; D=$S/baselines/sf1p3b; OUT=${1:?outdir}; IMG=${2:-$S/lingbot-world-v2/examples/03/image.jpg}; P=${3:-$S/lingbot-world-v2/examples/03}
mkdir -p $OUT; echo "[$(date +%F_%T)] waiting for /tmp/gpu6.lock" >> $OUT/smoke.log
flock /tmp/gpu6.lock env CUDA_VISIBLE_DEVICES=6 bash -c "
  echo \"[\$(date +%F_%T)] lock acquired\" >> $OUT/smoke.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 6 >> $OUT/smoke.log
  MODE=tf EXTRA=--tf-calib TIMEOUT=450 $D/run.sh $IMG $P 1234 $OUT/tf >> $OUT/smoke.log 2>&1; echo \"[\$(date +%F_%T)] tf done rc \$?\" >> $OUT/smoke.log
  MODE=uniform:4 TIMEOUT=360 $D/run.sh $IMG $P 1234 $OUT/uniform4 >> $OUT/smoke.log 2>&1; echo \"[\$(date +%F_%T)] uniform4 done rc \$?\" >> $OUT/smoke.log
"
echo "[$(date +%F_%T)] lock released (rc $?)" >> $OUT/smoke.log; echo SMOKE_END >> $OUT/smoke.log
