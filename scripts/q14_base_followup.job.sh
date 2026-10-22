#!/bin/bash
# GPU6 (Q12 후): hands 업링크 그리드 (Fast hands 영상은 Q13 이 만듦) -> UniPC 핸드오프 S01
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$SCRATCH/lingbot-world-v2/specwm_out
for i in $(seq 1 300); do [ -f $SCRATCH/exp/Q12_DONE ] && break; sleep 60; done
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
echo "########## HANDOFF UNIPC S01 ($(date)) ##########"
python -m specwm.handoff_unipc --ckpt $PWD/lingbot-world-base-cam --scene 01 --image $SCRATCH/lingbot-world-v2/examples/01/image.jpg \
  --action-dir $SCRATCH/rt2_action_01 --prompt-file $SCRATCH/p_01.txt --fast-video $R/floor/fast/S01/W18_sink6/fast_alone_s1234/video.mp4 \
  --sigmas 0.4,0.625,0.833 --res 1.0,0.5 --bits 8 --out $R/handoff_grid/S01_unipc
FV=$R/floor/fast/Shands/W18_sink6/fast_alone_s1234/video.mp4
for i in $(seq 1 300); do [ -f $FV ] && break; sleep 60; done
if [ -f $FV ]; then
  echo "########## HANDOFF GRID hands ($(date)) ##########"
  python -m specwm.handoff_grid --ckpt $PWD/lingbot-world-base-cam --scene hands --image $SCRATCH/lingbot-world-v2/examples/hands/image.jpg \
    --action-dir $SCRATCH/hands_action --prompt-file $SCRATCH/p_hands1.txt --fast-video $FV \
    --sigmas 0.4,0.625,0.833 --res 1.0,0.5 --bits 8 --steps 8 --out $R/handoff_grid/Shands
fi
touch $SCRATCH/exp/Q14_DONE; echo "[Q14_DONE] $(date)"
