#!/bin/bash
# score finished bench-43 clips with the Qwen3-VL judge every few minutes (vlm_clip_score skips clips already scored); stops when STOP exists
S=/home/hk28965/specwm; source $S/env.sh; cd $S; B=lingbot-world-v2/specwm_out/bench43
while [ ! -f $B/STOP_SCORING ]; do
  python scripts/vlm_clip_score.py paper/1012/vlm_scores_bench43.json "teacher=$B/*/uniform4_s*/video.mp4" "student=$B/*/uniform4_1p3b_s*/video.mp4" "sdvg07=$B/*/ir_-0.7_s*/video.mp4" "sdvg22=$B/*/ir_-2.2_s*/video.mp4" "mg2=$B/*/mg2_s*/video.mp4" "wp=$B/*/wp_s*/output.mp4" "ours832=$B/*/pc1_832_s*/cloud/spec.mp4" "ours704=$B/*/pc1_704_s*/cloud/spec.mp4" > $B/vlm_loop.txt 2>&1
  sleep 240
done
