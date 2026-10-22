#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
python -m specwm.bench_budget \
  --ckpt ./lingbot-world-v2-14b-causal-fast \
  --image examples/03/image.jpg --action examples/03 \
  --frames 81 --local-attn 18 \
  --weight-spec int8-g128 --weight-backend ours --kv-alpha 0.9 --tiles 2x2 \
  --save-videos --out specwm_out/vid_int8
