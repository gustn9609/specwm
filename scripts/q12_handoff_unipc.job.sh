#!/bin/bash
# GPU6: 업링크 핸드오프를 공식 UniPC ODE 로 재검증 (S05, sigma 3 x res 2 x bits 8)
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out
echo "########## HANDOFF UNIPC S05 ($(date)) ##########"
python -m specwm.handoff_unipc --ckpt $PWD/lingbot-world-base-cam --scene 05 --image $SCRATCH/lingbot-world-v2/examples/05/image.jpg \
  --action-dir $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --fast-video $R/floor/fast/S05/W18_sink6/fast_alone_s1234/video.mp4 \
  --sigmas 0.4,0.625,0.833 --res 1.0,0.5 --bits 8 --out $R/handoff_grid/S05_unipc
touch $SCRATCH/exp/Q12_DONE; echo "[Q12_DONE] $(date)"
