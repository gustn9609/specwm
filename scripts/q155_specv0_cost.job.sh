#!/bin/bash
# spec v0 cost sweep on the two failure tasks (seed 1234, every:2): steps {1,4,8}, 360p, no-CFG, prefix window 9. Two GPUs, tasks sequential.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; SD=1234
for V in "steps4|--steps 4" "ctx9|--ctx 9" "steps1|--steps 1" "guide1|--guide 1" "steps8|--steps 8" "360p|--max-area 184320" "360p_steps4_ctx9|--max-area 184320 --steps 4 --ctx 9"; do N=${V%%|*}; A=${V#*|}
 for TASK in sand_filling_up_jar sandcastle_wave_impact; do D=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0/$TASK/every_2_${N}_s$SD; [ -f $D/video.mp4 ] && continue
  echo "########## SPECV0 $TASK every:2 $N s$SD ($(date)) ##########"; python -m specwm.spec_v0 --task $TASK --seed $SD --policy every:2 $A --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk|Error|Traceback|OutOfMemory"; done; done
touch $SCRATCH/exp/Q155_DONE; echo "[Q155_DONE] $(date)"
