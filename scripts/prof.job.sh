#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
python -m specwm.profile_phases \
  --ckpt ./lingbot-world-v2-14b-causal-fast \
  --image examples/03/image.jpg --action examples/03 \
  --frames 81 --local-attn 18 \
  --warmup-chunks 2 --steady-from 3 --hook-blocks 3 \
  --kernel-trace --out specwm_out/profile
