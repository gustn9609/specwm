#!/bin/bash
# [10/17 Astra calibration] translation response: P3 at u_scale 0.30 and 0.45 (2x / 3x Astra's preset forward speed), one batch per hold, then DA3
S=/home/hk28965/specwm; C=$S/baselines/astra/calib; R=$S/scripts/envrun.sh
for u in 0.30 0.45; do
  echo "[chain] $(date) waiting for the GPU-5 lock (astra P3 u_scale $u)"
  flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 U_SCALE=$u $S/baselines/astra/run.sh batch $C/jobs_calib_P3_u$u.json $C/batch_P3_u$u.log > $C/batch_P3_u$u.stdout 2>&1
  echo "[chain] $(date) astra P3 u_scale $u exit $?"
done
echo "[chain] $(date) waiting for the lock (DA3)"
flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 $R python $S/baselines/astra/calib_da3.py $C/da3_calib.json \
  astra_P3_u0.30=$C/garden_1_P3_u0.30/video.mp4:$C/garden_1_P3_u0.30/camera.npz:20 \
  astra_P3_u0.45=$C/garden_1_P3_u0.45/video.mp4:$C/garden_1_P3_u0.45/camera.npz:20 > $C/da3_calib_P3_sweep.log 2>&1
echo "[chain] $(date) da3 exit $?"
echo CHAIN_DONE
