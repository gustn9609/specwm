#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD
O=$SCRATCH/lingbot-world-v2/specwm_out/ours_same

echo "########## CASE A: 기억 (재방문 궤적 161프레임, Base/Fast와 동일) ##########"
python -m specwm.bench_ours_same \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast --scene 00 \
  --action-dir $SCRATCH/rt2_action --frames 161 \
  --bits 6 --kv-bits 4 --tiles 2x2 --local-attn 18 --sink 6 \
  --out $O/caseA_revisit

echo "########## CASE B: 물리 (손 조작 81프레임, Base/Fast와 동일) ##########"
python -m specwm.bench_ours_same \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast --scene hands \
  --action-dir $SCRATCH/hands_action --frames 81 \
  --prompt-file $SCRATCH/p1.txt \
  --bits 6 --kv-bits 4 --tiles 2x2 --local-attn 18 --sink 6 \
  --out $O/caseB_hands
echo "########## DONE ##########"
