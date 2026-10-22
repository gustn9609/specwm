#!/bin/bash
true
export VBENCH_CACHE_DIR=$SCRATCH/vbench_cache; export TORCH_HOME=$SCRATCH/.torch_hub; export HF_HUB_OFFLINE=1
R=$SCRATCH/lingbot-world-v2/specwm_out
cd $SCRATCH
python vbench_eval.py --stage eval --out $SCRATCH/vbench_out5 --videos \
  base_S05_s16_seed4321=$SCRATCH/lingbot-world-v2/specwm_out/revisit/base_official_S05_s16_seed4321/video.mp4 \
  base_S05_s16_seed9999=$SCRATCH/lingbot-world-v2/specwm_out/revisit/base_official_S05_s16_seed9999/video.mp4 \
  base_prefix_S05_L9=$SCRATCH/lingbot-world-v2/specwm_out/revisit/base_prefix_S05_L9/video.mp4 \
  hands_causal_N3=$SCRATCH/lingbot-world-v2/specwm_out/hands/base_causal_N3/video.mp4 \
  hands_causal_N6=$SCRATCH/lingbot-world-v2/specwm_out/hands/base_causal_N6/video.mp4 \
  hands_causal_N12=$SCRATCH/lingbot-world-v2/specwm_out/hands/base_causal_N12/video.mp4
echo "[VBENCH_JOB_DONE] exit=$?"
