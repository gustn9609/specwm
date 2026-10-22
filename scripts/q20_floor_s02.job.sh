#!/bin/bash
# GPU4 (VR 데몬 유휴 -> 종료 후): Base 플로어 S02 시드 777/8888 + hands 2718 (Q19 가 마지막에 VR catch-up 재실행)
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; F=$SCRATCH/lingbot-world-v2/specwm_out/floor/base
run_one() { D=$F/S$1_s$2; [ -f $D/revisit.json ] && { echo "[skip] $D"; return; }
  echo "########## BASE 70-step S$1 seed $2 ($(date)) ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/$1/image.jpg \
    --action-path $4 --prompt-file $5 --frames $3 --seed $2 --out $D; }
run_one 02 777 161 $SCRATCH/rt2_action_02 $SCRATCH/p_02.txt
run_one 02 8888 161 $SCRATCH/rt2_action_02 $SCRATCH/p_02.txt
run_one hands 2718 81 $SCRATCH/hands_action $SCRATCH/p_hands1.txt
run_one 00 777 161 $SCRATCH/rt2_action_00 $SCRATCH/p_00.txt
touch $SCRATCH/exp/Q20_DONE; echo "[Q20_DONE] $(date)"
