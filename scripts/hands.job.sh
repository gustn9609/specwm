#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
A=$SCRATCH/hands_action
O=$SCRATCH/lingbot-world-v2/specwm_out/hands

for P in 1 2; do
  echo "########## FAST prompt$P ##########"
  cd $SCRATCH/lingbot-world-v2
  PYTHONPATH=$PWD python -m specwm.bench_revisit --model fast \
    --ckpt $PWD/lingbot-world-v2-14b-causal-fast --scene hands --steps 4 \
    --local-attn 18 --sink 6 --action-dir $A --prompt-file $SCRATCH/p$P.txt \
    --out $O/fast_p$P
done
for P in 1 2; do
  echo "########## BASE prompt$P ##########"
  cd $SCRATCH/lingbot-world
  mkdir -p examples/hands && cp $SCRATCH/hands_first.jpg examples/hands/image.jpg
  PYTHONPATH=$PWD python -m specwm.bench_revisit --model base \
    --ckpt $PWD/lingbot-world-base-cam --scene hands --steps 70 --guide 5.0 \
    --action-dir $A --prompt-file $SCRATCH/p$P.txt --out $O/base_p$P
done
echo "########## DONE ##########"
