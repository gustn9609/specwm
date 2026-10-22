#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/.inductor_cache
cd $SCRATCH/lingbot-world-v2
python -m specwm.bench_fusion \
  --ckpt ./lingbot-world-v2-14b-causal-fast \
  --image examples/03/image.jpg --action examples/03 \
  --frames 81 --local-attn 18 \
  --modes baseline compile --compile-mode default \
  --out specwm_out/fusion
