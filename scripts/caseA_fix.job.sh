#!/bin/bash
true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
# Base/Fast(bench_revisit.py)가 쓴 것과 정확히 같은 기본 프롬프트
cat > $SCRATCH/pA.txt << "EOF"
A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky.
EOF
python -m specwm.bench_ours_same \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast --scene 00 \
  --action-dir $SCRATCH/rt2_action --frames 161 \
  --prompt-file $SCRATCH/pA.txt \
  --bits 6 --kv-bits 4 --tiles 2x2 --local-attn 18 --sink 6 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/ours_same/caseA_revisit_fixed
