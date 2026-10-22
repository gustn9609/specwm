#!/bin/bash
# probe2: draft only (= Fast video), at node 1 every chunk evaluate all cheap-check presets (distance + time). env: GPUS, JOBS="task:seed ...", FRAMES(optional), OUTSUB(optional)
cd /home/hk28965/specwm; export SCRATCH=/home/hk28965/specwm; source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd lingbot-world-v2; export PYTHONPATH=$PWD
for J in $JOBS; do TASK=${J%:*}; SD=${J#*:}; D=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0/${OUTSUB:-$TASK}/probe2_s$SD; [ -f $D/video.mp4 ] && continue
  echo "########## SPECV0 $TASK probe2 s$SD ($(date)) ##########"
  CUDA_VISIBLE_DEVICES=$GPUS python -m specwm.spec_v0 --task $TASK --seed $SD --policy probe2:1 ${FRAMES:+--frames $FRAMES} --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk|Error|Traceback|OutOfMemory|line "; done
