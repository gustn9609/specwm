#!/bin/bash
# VLM-score new runs as they finish (skips clips already scored): low-res 448/384 (12 prompts x 3 seeds) and the chunk-wise arms on bench-43
S=/home/hk28965/specwm; source $S/env.sh; cd $S; F=lingbot-world-v2/specwm_out/fused; B=lingbot-world-v2/specwm_out/bench43
while [ ! -f $B/STOP_SCORING2 ]; do
  python scripts/vlm_clip_score.py paper/1012/vlm_scores_diag.json "dg448_cloud=$F/*/dg448_s*/cloud/spec.mp4" "dg384_cloud=$F/*/dg384_s*/cloud/spec.mp4" > $B/vlm_lowres.txt 2>&1
  python scripts/vlm_clip_score.py paper/1012/vlm_scores_bench43.json "cw07_sep=$B/*/cw07_sep_s*/video.mp4" "cw07_pc1=$B/*/cw07_pc1_s*/video.mp4" "cw07_safe=$B/*/cw07_safe_s*/video.mp4" "cw05_pc1=$B/*/cw05_pc1_s*/video.mp4" > $B/vlm_cw.txt 2>&1
  sleep 300
done
