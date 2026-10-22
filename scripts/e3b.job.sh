#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out
python -m specwm.resync_v2_e3b --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/rt2_action_05 \
  --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 --seeds 1234,4321,9999 \
  --base-video $R/revisit/base_official_S05/video.mp4 --base16-video $R/revisit/base_official_S05_s16/video.mp4 \
  --prefix-video $R/revisit/base_prefix_S05_L9/video.mp4 --out $R/resync_v2
