#!/bin/bash
# GPU6 (Q17 후): Base 플로어 S01 seed 8888
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 1 480); do [ -f $SCRATCH/exp/Q17_DONE ] && break; sleep 60; done
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; F=$SCRATCH/lingbot-world-v2/specwm_out/floor/base
D=$F/S01_s8888; if [ ! -f $D/revisit.json ]; then echo "########## BASE 70-step S01 seed 8888 ($(date)) ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/01/image.jpg \
    --action-path $SCRATCH/rt2_action_01 --prompt-file $SCRATCH/p_01.txt --frames 161 --seed 8888 --out $D; fi
touch $SCRATCH/exp/Q23_DONE; echo "[Q23_DONE] $(date)"
