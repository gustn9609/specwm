#!/bin/bash
true
source $SCRATCH/venv/bin/activate
cd $SCRATCH/lingbot-world-v2
export PYTHONPATH=$PWD:$PYTHONPATH
A=$SCRATCH/rt2_action
O=$SCRATCH/lingbot-world-v2/specwm_out/revisit

echo "########## FAST (4 steps) ##########"
python -m specwm.bench_revisit --model fast \
  --ckpt $SCRATCH/ckpt/LingBot-World-Fast --scene 00 --steps 4 \
  --local-attn 18 --sink 6 --action-dir $A --out $O/fast_rt2

echo "########## BASE (70 steps, bidirectional) ##########"
python -m specwm.bench_revisit --model base \
  --ckpt $SCRATCH/ckpt/LingBot-World --scene 00 --steps 70 \
  --guide 5.0 --action-dir $A --out $O/base_rt2
echo "########## DONE ##########"
