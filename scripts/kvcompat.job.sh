#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
python -m specwm.kv_compat \
  --fast-ckpt $PWD/lingbot-world-v2-14b-causal-fast \
  --base-ckpt $SCRATCH/lingbot-world/lingbot-world-base-cam \
  --action-root $SCRATCH/rt2_action_00 \
  --prompt-file $SCRATCH/p_00.txt --frames 13 --max-layers 8 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/kv_compat
