#!/bin/bash
# warm-up + gating: first W chunks always corrected, then node-1 check (thr = probe median of that seed), run-length R; correction = Base RePaint 4-step 480p.
# env: GPUS, TASKS, SEEDS, TAG
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm
source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd lingbot-world-v2; export PYTHONPATH=$PWD
for SD in $SEEDS; do for TASK in $TASKS; do
 [ -f $SCRATCH/lingbot-world-v2/specwm_out/spec_v0/$TASK/probe_s$SD/spec_v0.json ] || { echo "[skip $TASK s$SD: no probe]"; continue; }
 THR=$($SCRATCH/venv/bin/python $SCRATCH/scripts/spec_v0_pick.py thr $TASK $SD 1 0.5)
 for WR in "2 2" "3 2" "2 1"; do set -- $WR; N=w$1r$2; D=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0/$TASK/${N}_s$SD; [ -f $D/video.mp4 ] && continue
  echo "########## SPECV0 $TASK $N thr $THR s$SD ($(date)) ##########"
  CUDA_VISIBLE_DEVICES=$GPUS python -m specwm.spec_v0 --task $TASK --seed $SD --policy verify_node:1:$THR --warmup $1 --run $2 --steps 4 --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk|Error|Traceback|OutOfMemory"; done; done; done
touch $SCRATCH/exp/Q162_${TAG}_DONE; echo "[Q162_DONE $TAG] $(date)"
