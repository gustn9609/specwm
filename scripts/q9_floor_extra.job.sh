#!/bin/bash
# GPU6 (Q2/Q8 종료 후): Base 플로어 추가 시드 (S05/S01 seed 2718 -> 씬당 4시드=6쌍), 그 다음 hands
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out; F=$R/floor/base
run_one() { D=$F/S$1_s$2; [ -f $D/revisit.json ] && { echo "[skip] $D"; return; }
  echo "########## BASE 70-step S$1 seed $2 ($(date)) ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/$1/image.jpg \
    --action-path $4 --prompt-file $5 --frames $3 --seed $2 --out $D; }
run_one 05 2718 161 $SCRATCH/rt2_action_05 $SCRATCH/p_05.txt
run_one 01 2718 161 $SCRATCH/rt2_action_01 $SCRATCH/p_01.txt
run_one hands 1234 81 $SCRATCH/hands_action $SCRATCH/p_hands1.txt
run_one hands 4321 81 $SCRATCH/hands_action $SCRATCH/p_hands1.txt
touch $SCRATCH/exp/Q9_DONE; echo "[Q9_DONE] $(date)"
