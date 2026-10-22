#!/bin/bash
# sweep round 2 (after night B on 4+5): combos without the ctx window (which color-drifts): 360p+4step, 4step+noCFG, 360p+4step+noCFG. seed 1234, every:2.
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm; while [ ! -f exp/Q156_B_DONE ]; do sleep 120; done
source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd lingbot-world-v2; export PYTHONPATH=$PWD; SD=1234
for V in "360p_steps4|--max-area 184320 --steps 4" "steps4_guide1|--steps 4 --guide 1" "360p_steps4_guide1|--max-area 184320 --steps 4 --guide 1"; do N=${V%%|*}; A=${V#*|}
 for TASK in sand_filling_up_jar sandcastle_wave_impact; do D=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0/$TASK/every_2_${N}_s$SD; [ -f $D/video.mp4 ] && continue
  echo "########## SPECV0 $TASK every:2 $N s$SD ($(date)) ##########"; CUDA_VISIBLE_DEVICES=4,5 python -m specwm.spec_v0 --task $TASK --seed $SD --policy every:2 $A --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk|Error|Traceback|OutOfMemory"; done; done
touch $SCRATCH/exp/Q160_DONE; echo "[Q160_DONE] $(date)"
