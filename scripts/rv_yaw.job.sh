#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
for st in 4; do
  python -m specwm.bench_revisit --model fast --ckpt ./lingbot-world-v2-14b-causal-fast \
    --scene 00 --steps $st --local-attn 18 --action-dir specwm_out/revisit/yaw_action --out specwm_out/revisit/fast_yaw
done
