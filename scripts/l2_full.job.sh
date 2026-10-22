#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2
# L1' today (fast_bf16 target). When causal-pretrain drops:
#   --target causal_pretrain --target-ckpt <dir> --target-substeps 10
python -m specwm.l2_spec_chunk --ckpt ./lingbot-world-v2-14b-causal-fast \
  --image examples/03/image.jpg --action examples/03 \
  --target fast_bf16 --frames 161 --quant int8 --local-attn 12
