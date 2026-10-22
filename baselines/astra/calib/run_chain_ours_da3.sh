#!/bin/bash
# [10/17 Astra calibration] chain, each step its own GPU-5 lock hold (< 15 min each):
#  1. OUR teacher (14B FP8, 4-step, solo) on garden_1 (stretched to 832x480 like Astra's loader) for P3 and P1 at Astra's area 399360
#     -> ours_teacher/calib_garden_1_P3/spec.mp4 (16 fps, 161 frames)   = the reference for "P3's forward/turn ratio matches ours"
#  2. DA3 metric pose recovery (calib_da3.py) of Astra P1 (u_scale 0.15), Astra P2 (LATENTS=56), ours P3, ours P1 -> da3_calib.json
S=/home/hk28965/specwm; C=$S/baselines/astra/calib; R=$S/scripts/envrun.sh
echo "[chain] $(date) waiting for the GPU-5 lock (ours)"
flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 SPECWM_TASKS=$C/ours_tasks/tasks.json $R python -m specwm.two_machine --role solo --out $C/ours_teacher --wq fp8 \
  --nodes 0,250,500,750 --use 0,1,2,3 --tasks calib_garden_1_P3,calib_garden_1_P1 --area 399360 --window 18 > $C/ours_teacher.log 2>&1
echo "[chain] $(date) ours exit $? ; waiting for the lock (DA3)"
flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 $R python $S/baselines/astra/calib_da3.py $C/da3_calib.json \
  astra_P1_u0.15=$C/garden_1_P1_u0.15/video.mp4:$C/garden_1_P1_u0.15/camera.npz:20 \
  astra_P2_lat56=$S/baselines/runs/astra/astra__garden_1__P2/video.mp4:$S/baselines/runs/astra/astra__garden_1__P2/camera.npz:20 \
  ours_P3=$C/ours_teacher/calib_garden_1_P3/spec.mp4:$C/ours_tasks/calib_garden_1_P3/poses.npy:16 \
  ours_P1=$C/ours_teacher/calib_garden_1_P1/spec.mp4:$C/ours_tasks/calib_garden_1_P1/poses.npy:16 > $C/da3_calib.log 2>&1
echo "[chain] $(date) da3 exit $?"
echo CHAIN_DONE
