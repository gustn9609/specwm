#!/bin/bash
# AR pair #2 on S05 seeds (360p): draft = 1.3B causal-fast, target = Fast 14B (4-step). Modes: fast_only (1.3B alone) / periodic:3 / periodic:2 / verify:999. One GPU.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; O=$SCRATCH/lingbot-world-v2/specwm_out/spec_ar_1p3b/S05; mkdir -p $O
for SD in $SEEDS; do for M in fast_only periodic:3 periodic:2 verify:999; do D=$O/${M/:/_}_s$SD; [ -f $D/video.mp4 ] && continue
  echo "########## SPEC_AR_1P3B $M s$SD ($(date)) ##########"; python -m specwm.spec_ar --scene 05 --seed $SD --mode $M --draft fast1p3b --target fast14b --out $D 2>&1 | grep -E "^\[SPEC|Error|Traceback|OutOfMemory"; done; done
touch $SCRATCH/exp/Q148_DONE; echo "[Q148_DONE] $(date)"
