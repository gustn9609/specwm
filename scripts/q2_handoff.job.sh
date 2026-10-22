#!/bin/bash
# Q2: Pyramid-Flow 식 업링크 그리드 (action item #4). Base 정제 1스텝 ~41s -> 6h 예산:
#   S05: 전체 그리드 (sigma 5 x res 3 x bits 2 + UPONLY 6 = 36 arm, K=8, ~3.5h)   S01/S00/S02: 축소 그리드 (~1h 씩)
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out
run_grid() {  # SC SIGMAS RES BITS STEPS
  FV=$R/floor/fast/S$1/W18_sink6/fast_alone_s1234/video.mp4
  for i in $(seq 1 360); do [ -f $FV ] && [ -f $R/floor/fast/S$1/resync_v2.json ] && break; sleep 60; done
  [ -f $FV ] || { echo "[Q2] S$1 fast video 없음, 건너뜀"; return; }
  echo "########## HANDOFF GRID S$1 ($(date)) sig=$2 res=$3 bits=$4 K=$5 ##########"
  python -m specwm.handoff_grid --ckpt $PWD/lingbot-world-base-cam --scene $1 --image $SCRATCH/lingbot-world-v2/examples/$1/image.jpg \
    --action-dir $SCRATCH/rt2_action_$1 --prompt-file $SCRATCH/p_$1.txt --fast-video $FV \
    --sigmas $2 --res $3 --bits $4 --steps $5 --out $R/handoff_grid/S$1
}
run_grid 05 0.2,0.4,0.625,0.833,0.937 1.0,0.75,0.5 16,8 8
touch $SCRATCH/exp/Q2_STAGE1_DONE
run_grid 01 0.4,0.625,0.833 1.0,0.5 8 8
run_grid 00 0.4,0.625,0.833 1.0,0.5 8 8
run_grid 02 0.4,0.625,0.833 1.0,0.5 8 8
run_grid 01 0.2,0.937 1.0,0.75,0.5 16,8 8
touch $SCRATCH/exp/Q2_DONE; echo "[Q2_DONE] $(date)"
