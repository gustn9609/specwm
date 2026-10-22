#!/bin/bash
true
export VBENCH_CACHE_DIR=$SCRATCH/vbench_cache; export TORCH_HOME=$SCRATCH/.torch_hub; export HF_HUB_OFFLINE=1
R=$SCRATCH/lingbot-world-v2/specwm_out
cd $SCRATCH
python vbench_eval.py --stage eval --out $SCRATCH/vbench_out --videos \
  fast_S05_memory=$R/multi_scene/S05_W18_bf16_s1234/video.mp4 \
  ours_S00_memory=$R/ours_same/caseA_revisit_fixed/OURS_gptq6_kv4_tiled/video.mp4 \
  base_hands=$R/hands/base_p1/video.mp4 \
  fast_hands=$R/hands/fast_p1/video.mp4 \
  ours_hands=$R/ours_same/caseB_hands/OURS_gptq6_kv4_tiled/video.mp4 \
  base_S05_buggy_sampler=$R/revisit/base_S05/video.mp4
echo "[VBENCH_JOB_DONE] exit=$?"
