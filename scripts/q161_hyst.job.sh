#!/bin/bash
# gated correction with hysteresis / run-length: once a chunk is rejected at node 1, correct the next RUN-1 chunks too. seed 1234, both tasks, 4-step 480p correction.
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm; 
source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd lingbot-world-v2; export PYTHONPATH=$PWD; SD=1234
for TASK in sand_filling_up_jar sandcastle_wave_impact; do
 THR=$($SCRATCH/venv/bin/python $SCRATCH/scripts/spec_v0_pick.py thr $TASK $SD 1 0.5)
 for RUN in 2 3; do D=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0/$TASK/vnode1run${RUN}_s$SD; [ -f $D/video.mp4 ] && continue
  echo "########## SPECV0 $TASK vnode1run$RUN thr $THR s$SD ($(date)) ##########"
  CUDA_VISIBLE_DEVICES=4,5 python -m specwm.spec_v0 --task $TASK --seed $SD --policy verify_node:1:$THR --run $RUN --steps 4 --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk|Error|Traceback|OutOfMemory"; done; done
touch $SCRATCH/exp/Q161_DONE; echo "[Q161_DONE] $(date)"
