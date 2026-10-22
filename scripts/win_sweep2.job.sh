#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
python -m specwm.win_sweep2 \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast --scene 00 \
  --action-dir $SCRATCH/rt2_action --prompt-file $SCRATCH/pA.txt \
  --frames 161 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/win_sweep2
