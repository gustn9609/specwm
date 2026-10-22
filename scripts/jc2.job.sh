#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out
python -m specwm.resync_v3 --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/traj20_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 321 \
  --seeds 1234 --base-video $R/revisit/base_official_S05/video.mp4 --configs W18_sink6,W24_sink18 --out $R/resync20
