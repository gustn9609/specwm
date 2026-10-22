#!/bin/bash
true
source $SCRATCH/venv/bin/activate
A=$SCRATCH/rt2_action
O=$SCRATCH/lingbot-world-v2/specwm_out/revisit

echo "########## FAST (v2 repo, 4 steps) ##########"
cd $SCRATCH/lingbot-world-v2
PYTHONPATH=$PWD python -m specwm.bench_revisit --model fast \
  --ckpt $PWD/lingbot-world-v2-14b-causal-fast --scene 00 --steps 4 \
  --local-attn 18 --sink 6 --action-dir $A --out $O/fast_rt2

echo "########## BASE (v1 repo, 70 steps, bidirectional) ##########"
cd $SCRATCH/lingbot-world
PYTHONPATH=$PWD python -m specwm.bench_revisit --model base \
  --ckpt $PWD/lingbot-world-base-cam --scene 00 --steps 70 \
  --guide 5.0 --action-dir $A --out $O/base_rt2
echo "########## DONE ##########"
