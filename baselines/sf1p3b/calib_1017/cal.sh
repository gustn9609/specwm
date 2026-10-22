#!/bin/bash
# [baselines/sf1p3b/calib_1017] threshold calibration of the training-free verifier on the 1.3B's OWN 4-step output (arm B of the gap list):
# 3 neutral scenes x programs P1, P2 (tasks.json built by make_task.py), one process, --tf-calib (untimed dense 4-step reference per chunk ->
# oracle gap rel_k vs the smoothed scores) with --tau-red -1 --tau-stab -1 so that NO token is accepted: the committed context of every chunk is
# then exactly the uniform:4 output (same code path: all tokens forwarded through fast_patch.ACTIVE at steps 2-4), i.e. the scores are measured
# along the model's own full-step rollout as in the paper's calibration, not along a collapsing early-exit rollout (smoke_1016 did the latter).
# GPU 5 under flock /tmp/gpu5.lock, hard timeout 14 min on the python process. usage: cal.sh   (outputs: raw/<task>/sf1p3b_tfcal_s1234/{loop.json,calib.pt,video.mp4})
S=/home/hk28965/specwm; C=$S/baselines/sf1p3b/calib_1017; R=$S/scripts/envrun.sh
TASKS=$(cd $C && $R python -c "import json;print(','.join(t['id'] for t in json.load(open('$C/tasks.json'))))")
echo "[$(date +%F_%T)] waiting for /tmp/gpu5.lock; tasks $TASKS" >> $C/cal.log
flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 bash -c "
  echo \"[\$(date +%F_%T)] lock acquired\" >> $C/cal.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5 >> $C/cal.log
  SAVE_LAT=0 timeout -k 20 840 $R python -m specwm.spec_early_tf --model 1p3b --mode tf --tasks $TASKS --seeds 1234 --out $C/raw --tasks-json $C/tasks.json --prompt '' \
     --area 399360 --frames 161 --tau-red -1 --tau-stab -1 --tf-calib --bucket 128 --warm 1 --stream-decode 0 --arm sf1p3b_tfcal >> $C/cal.log 2>&1
  echo \"[\$(date +%F_%T)] calib done rc \$?\" >> $C/cal.log
"
echo "[$(date +%F_%T)] lock released" >> $C/cal.log; echo CAL_END >> $C/cal.log
