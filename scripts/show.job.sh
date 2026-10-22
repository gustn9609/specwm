#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
python -m specwm.bench_showdown \
  --ckpt ./lingbot-world-v2-14b-causal-fast \
  --image examples/03/image.jpg --action examples/03 \
  --frames 81 --local-attn 18 \
  --bits 6 5 --tiles 2x2 --out specwm_out/showdown
