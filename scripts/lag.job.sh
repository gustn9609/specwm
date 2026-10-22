#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
# expandable_segments failed to map pages despite 11 GiB free -> disabled
cd $SCRATCH/lingbot-world-v2
python -m specwm.lag_sweep \
  --ckpt ./lingbot-world-v2-14b-causal-fast \
  --image examples/03/image.jpg --action examples/03 \
  --frames 81 --local-attn 18 --tile-mode quad \
  --lags 0 1 2 3 4 --trees 2 4 --out specwm_out/lagsweep
