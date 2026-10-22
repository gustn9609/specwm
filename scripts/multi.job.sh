#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
python -m specwm.bench_multi --ckpt ./lingbot-world-v2-14b-causal-fast \
  --scenes 00 01 02 --frames 161 --local-attn 18 --bits 6 --tiles 2x2 \
  --out specwm_out/multi
