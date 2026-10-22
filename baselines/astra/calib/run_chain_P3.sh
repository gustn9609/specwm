#!/bin/bash
# [10/17 Astra calibration] verification: Astra P3 (forward 1 u/s + yaw 12 deg/s, u_scale 0.15) on garden_1 -> runs/astra/astra__garden_1__P3, then DA3 on it
S=/home/hk28965/specwm; C=$S/baselines/astra/calib; R=$S/scripts/envrun.sh
echo "[chain] $(date) waiting for the GPU-5 lock (astra P3)"
flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 $S/baselines/astra/run.sh batch $C/jobs_calib_P3.json $C/batch_P3.log > $C/batch_P3.stdout 2>&1
echo "[chain] $(date) astra P3 exit $? ; waiting for the lock (DA3)"
flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 $R python $S/baselines/astra/calib_da3.py $C/da3_calib.json \
  astra_P3_u0.15=$S/baselines/runs/astra/astra__garden_1__P3/video.mp4:$S/baselines/runs/astra/astra__garden_1__P3/camera.npz:20 > $C/da3_calib_P3.log 2>&1
echo "[chain] $(date) da3 exit $?"
echo CHAIN_DONE
