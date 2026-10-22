#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world
export PYTHONPATH=$PWD
for SC in 01 05; do
  echo "########## BASE scene $SC ##########"
  mkdir -p examples/$SC
  cp $SCRATCH/lingbot-world-v2/examples/$SC/image.jpg examples/$SC/ 2>/dev/null
  python -m specwm.bench_revisit --model base \
    --ckpt $PWD/lingbot-world-base-cam --scene $SC --steps 70 --guide 5.0 \
    --action-dir $SCRATCH/rt2_action_$SC --prompt-file $SCRATCH/p_$SC.txt \
    --out $SCRATCH/lingbot-world-v2/specwm_out/revisit/base_S$SC
done
