#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2
python -m specwm.l2_compare --ckpt ./lingbot-world-v2-14b-causal-fast   --image examples/03/image.jpg --action examples/03 --frames 81   --local-attn 18 --modes target --no-draft-model --out specwm_out/full_target
python -m specwm.l2_compare --ckpt ./lingbot-world-v2-14b-causal-fast   --image examples/03/image.jpg --action examples/03 --frames 81   --local-attn 18 --quant int8-g128 --modes draft --out specwm_out/full_draft
