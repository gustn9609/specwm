#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world
export PYTHONPATH=$PWD
python -m specwm.hybrid_refine \
  --ckpt $PWD/lingbot-world-base-cam --scene hands \
  --action-dir $SCRATCH/hands_action --prompt-file $SCRATCH/p1.txt \
  --fast-latent $SCRATCH/lingbot-world-v2/specwm_out/hands/fast_p1/latent.pt \
  --frames 81 --guide 5.0 \
  --schedules "0.25;0.15;0.25,0.10;0.40,0.20,0.08" \
  --out $SCRATCH/lingbot-world-v2/specwm_out/hybrid/hands
