#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
python -m specwm.multi_scene \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast \
  --action-root $SCRATCH --prompt-root $SCRATCH --frames 161 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/multi_scene
