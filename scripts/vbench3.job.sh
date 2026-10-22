#!/bin/bash
true
export VBENCH_CACHE_DIR=$SCRATCH/vbench_cache; export TORCH_HOME=$SCRATCH/.torch_hub; export HF_HUB_OFFLINE=1
R=$SCRATCH/lingbot-world-v2/specwm_out
cd $SCRATCH
python vbench_eval.py --stage eval --out $SCRATCH/vbench_out3 --videos \
  resync_fast_alone=$SCRATCH/lingbot-world-v2/specwm_out/resync_s05/fast_alone/video.mp4 \
  resync_ctrl_k6=$SCRATCH/lingbot-world-v2/specwm_out/resync_s05/ctrl_selfcommit_k6/video.mp4 \
  resync_k3=$SCRATCH/lingbot-world-v2/specwm_out/resync_s05/resync_k3/video.mp4 \
  resync_k6=$SCRATCH/lingbot-world-v2/specwm_out/resync_s05/resync_k6/video.mp4 \
  resync_k9=$SCRATCH/lingbot-world-v2/specwm_out/resync_s05/resync_k9/video.mp4 \
  resync_every3=$SCRATCH/lingbot-world-v2/specwm_out/resync_s05/resync_every3/video.mp4 \
  base_S05_s4=$SCRATCH/lingbot-world-v2/specwm_out/revisit/base_official_S05_s4/video.mp4 \
  base_S05_s8=$SCRATCH/lingbot-world-v2/specwm_out/revisit/base_official_S05_s8/video.mp4 \
  base_S05_s16=$SCRATCH/lingbot-world-v2/specwm_out/revisit/base_official_S05_s16/video.mp4 \
  base_hands_s4=$SCRATCH/lingbot-world-v2/specwm_out/hands/base_official_p1_s4/video.mp4 \
  base_hands_s16=$SCRATCH/lingbot-world-v2/specwm_out/hands/base_official_p1_s16/video.mp4 \
  solar_stage1_nfe50=$SCRATCH/solar_stage1_nfe50.mp4 \
  solar_stage1_nfe4=$SCRATCH/solar_stage1_nfe4.mp4 \
  solar_stage2_nfe4=$SCRATCH/solar_stage2_nfe4.mp4

echo "[VBENCH_JOB_DONE] exit=$?"
