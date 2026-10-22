#!/bin/bash
# Q2 종료 후 GPU6: cloud 정제 스텝 수 K 축소 (K=4, K=2) — 업링크 그리드 핵심 부분집합 (S05)
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out
for i in $(seq 1 300); do [ -f $SCRATCH/exp/Q2_DONE ] && break; sleep 60; done
FV=$R/floor/fast/S05/W18_sink6/fast_alone_s1234/video.mp4
for K in 4 2; do
  echo "########## HANDOFF GRID S05 K=$K ($(date)) ##########"
  python -m specwm.handoff_grid --ckpt $PWD/lingbot-world-base-cam --scene 05 --image $SCRATCH/lingbot-world-v2/examples/05/image.jpg \
    --action-dir $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --fast-video $FV \
    --sigmas 0.4,0.625,0.833 --res 1.0,0.5 --bits 8 --steps $K --out $R/handoff_grid/S05_K$K
done
touch $SCRATCH/exp/Q8_DONE; echo "[Q8_DONE] $(date)"
