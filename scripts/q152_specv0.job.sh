#!/bin/bash
# Spec-decoding v0 on the two failure tasks (480p, 161 f, 16-step Base RePaint correction). TASK, SEEDS env. Two GPUs.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; O=$SCRATCH/lingbot-world-v2/specwm_out/spec_v0/$TASK; mkdir -p $O
for SD in $SEEDS; do for P in never verify:999 every:3 every:2 always; do D=$O/${P/:/_}_s$SD; [ -f $D/video.mp4 ] && continue
  echo "########## SPECV0 $TASK $P s$SD ($(date)) ##########"; python -m specwm.spec_v0 --task $TASK --seed $SD --policy $P --out $D 2>&1 | grep -E "^\[SPEC|^\[chunk|Error|Traceback|OutOfMemory"; done; done
touch $SCRATCH/exp/Q152_${TASK}_DONE; echo "[Q152_DONE $TASK] $(date)"
