#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
python -m specwm.win60 \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast \
  --action-root $SCRATCH/traj60_action \
  --prompt-file $SCRATCH/p_00.txt --frames 893 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/win60
