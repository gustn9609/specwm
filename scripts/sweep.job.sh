#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world
python -m specwm.l1_draft_sweep --ckpt ./lingbot-world-base-cam   --image examples/00/image.jpg --action examples/00 --nodes 20 --seeds 2 --specs nf4 nf4-g64 --out specwm_out/draft_sweep_nf4
