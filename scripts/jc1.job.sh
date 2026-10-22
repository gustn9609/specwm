#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out
python -m specwm.resync_v3 --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/rt2_action_05 \
  --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 --seeds 1234 --configs W18_sink6 --xfade --prefix-videos sep_self=$SCRATCH/lingbot-world-v2/specwm_out/revisit/e3sep_self_mask/video.mp4,sep_other=$SCRATCH/lingbot-world-v2/specwm_out/revisit/e3sep_otherseed_mask/video.mp4,repaint_fast=$SCRATCH/lingbot-world-v2/specwm_out/revisit/e3a_repaint_fast/video.mp4,repaint_self=$SCRATCH/lingbot-world-v2/specwm_out/revisit/e3a_repaint_self/video.mp4,causal12=$SCRATCH/lingbot-world-v2/specwm_out/revisit/e3b_causal12_repaint_fast/video.mp4,laundered=$SCRATCH/lingbot-world-v2/specwm_out/revisit/e3c_repaint_laundered/video.mp4 \
  --base-video $R/revisit/base_official_S05/video.mp4 --base16-video $R/revisit/base_official_S05_s16/video.mp4 \
  --out $R/resync_v2
