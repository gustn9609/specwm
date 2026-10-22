#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world
python -m specwm.l1_tile_stats --ckpt ./lingbot-world-base-cam   --image examples/00/image.jpg --action examples/00
