#!/bin/bash
# Q1: Base 70-step 노이즈 플로어 (action item #2). spi0 에서 런당 ~45분 -> 6h 예산에 맞춰 우선순위:
#   (1) S05/S01 시드 4321, 9999 (기존 1234 와 합쳐 씬당 3시드 = 3쌍)  (2) S00/S02 시드 1234, 4321  (3) 여유 시 hands / 5시드
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out; F=$R/floor/base; mkdir -p $F
for SC in 05 01; do [ -e $F/S${SC}_s1234 ] || ln -s $R/revisit/base_official_S$SC $F/S${SC}_s1234; done
run_one() {  # SC SEED FRAMES ACTION PROMPT
  D=$F/S$1_s$2; [ -f $D/revisit.json ] && { echo "[skip] $D"; return; }
  echo "########## BASE 70-step S$1 seed $2 ($(date)) ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/$1/image.jpg \
    --action-path $4 --prompt-file $5 --frames $3 --seed $2 --out $D
}
for SEED in 4321 9999; do for SC in 05 01; do run_one $SC $SEED 161 $SCRATCH/rt2_action_$SC $SCRATCH/p_$SC.txt; done; done
touch $SCRATCH/exp/Q1_STAGE1_DONE
for SEED in 1234 4321; do for SC in 00 02; do run_one $SC $SEED 161 $SCRATCH/rt2_action_$SC $SCRATCH/p_$SC.txt; done; done
touch $SCRATCH/exp/Q1_STAGE2_DONE
for SEED in 1234 4321 9999; do run_one hands $SEED 81 $SCRATCH/hands_action $SCRATCH/p_hands1.txt; done
for SEED in 2718 31415; do for SC in 05 01 00 02; do run_one $SC $SEED 161 $SCRATCH/rt2_action_$SC $SCRATCH/p_$SC.txt; done; done
for SC in 00 02; do run_one $SC 9999 161 $SCRATCH/rt2_action_$SC $SCRATCH/p_$SC.txt; done
touch $SCRATCH/exp/Q1_DONE; echo "[Q1_DONE] $(date)"
