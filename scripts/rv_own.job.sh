#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world
python -m specwm.bench_revisit --model base --ckpt ./lingbot-world-base-cam \
  --scene 00 --scene-root examples --steps 70 --frames 81 --action-dir examples/00 --prompt-file examples/00/prompt.txt \
  --out $SCRATCH/lingbot-world-v2/specwm_out/revisit/base_own70
