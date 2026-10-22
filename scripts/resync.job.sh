#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
python -m specwm.resync --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/rt2_action_05 \
  --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 \
  --base-video $SCRATCH/lingbot-world-v2/specwm_out/revisit/base_official_S05/video.mp4 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/resync_s05
