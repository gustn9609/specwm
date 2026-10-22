#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
PYTHONPATH=$PWD python -m specwm.viz_denoise \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast --scene 00 --steps 4 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/viz_denoise
