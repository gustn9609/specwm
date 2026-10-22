#!/bin/bash
# w2r2 gating (warm-up 2, run 2, 4-step 480p correction) with a cheap check preset; thr = median of that preset in the probe2 run of the same seed. env: GPUS, JOBS="task:seed ...", PRESETS
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm; source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd lingbot-world-v2; export PYTHONPATH=$PWD
for P in $PRESETS; do for J in $JOBS; do TASK=${J%:*}; SD=${J#*:}; V=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0/$TASK
  [ -f $V/probe2_s$SD/spec_v0.json ] || { echo "[skip $TASK s$SD $P: no probe2]"; continue; }
  N=w2r2_$P; [ $P = full ] && N=w2r2; D=$V/${N}_s$SD; [ -f $D/video.mp4 ] && continue
  THR=$($SCRATCH/venv/bin/python $SCRATCH/scripts/spec_v0_pick.py thr2 $TASK $SD $P 0.5)
  echo "########## SPECV0 $TASK $N thr $THR s$SD ($(date)) ##########"
  CUDA_VISIBLE_DEVICES=$GPUS python -m specwm.spec_v0 --task $TASK --seed $SD --policy verify_node:1:$THR --chk $P --warmup 2 --run 2 --steps 4 --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk|Error|Traceback|OutOfMemory"; done; done
