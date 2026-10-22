#!/bin/bash
true
export VBENCH_CACHE_DIR=$SCRATCH/vbench_cache; export TORCH_HOME=$SCRATCH/.torch_hub; export HF_HUB_OFFLINE=1
R=$SCRATCH/lingbot-world-v2/specwm_out
cd $SCRATCH
python vbench_eval.py --stage eval --out $SCRATCH/vbench_out2 --videos \
  base_S05_official=$R/revisit/base_official_S05/video.mp4 \
  base_S01_official=$R/revisit/base_official_S01/video.mp4 \
  base_hands_official=$R/hands/base_official_p1/video.mp4
echo "[VBENCH_JOB_DONE] exit=$?"
