#!/bin/bash
# AR speculative pair on S05 seeds (360p): fast_only / pretrain_only (target distribution) / periodic:3 / periodic:2 / verify:999 (accept-all, logs x0 distances). Needs 2 GPUs (CUDA_VISIBLE_DEVICES=a,b).
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; O=$SCRATCH/lingbot-world-v2/specwm_out/spec_ar/S05; mkdir -p $O
for SD in $SEEDS; do for M in fast_only pretrain_only periodic:3 periodic:2 verify:999; do D=$O/${M/:/_}_s$SD; [ -f $D/video.mp4 ] && continue
  echo "########## SPEC_AR $M s$SD ($(date)) ##########"; python -m specwm.spec_ar --scene 05 --seed $SD --mode $M --out $D 2>&1 | grep -E "^\[SPEC|Error|Traceback|OutOfMemory"; done; done
touch $SCRATCH/exp/Q147_${SHARD}_DONE; echo "[Q147_DONE shard $SHARD] $(date)"
