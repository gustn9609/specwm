#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
python -m specwm.bench_bits \
  --ckpt ./lingbot-world-v2-14b-causal-fast \
  --image examples/03/image.jpg --action examples/03 \
  --frames 81 --local-attn 18 \
  --skip-naive --out specwm_out/bits_gptq
