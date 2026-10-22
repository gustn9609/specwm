#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2
python -m specwm.l2_compare --ckpt ./lingbot-world-v2-14b-causal-fast   --image examples/03/image.jpg --action examples/03   --frames 81 --quant int8-g128 --local-attn 12 --out specwm_out/compare_la12
