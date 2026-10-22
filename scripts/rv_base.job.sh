#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world
python -m specwm.bench_revisit --model base --ckpt ./lingbot-world-base-cam \
  --scene 00 --scene-root ../lingbot-world-v2/examples --steps 40 --half 80 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/revisit/base_s40
