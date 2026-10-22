#!/bin/bash
# when the uniform 3-step bench-43 runs finish: VLM-score them (same judge/protocol as bench-43)
S=/home/hk28965/specwm; B=$S/lingbot-world-v2/specwm_out/bench43; source $S/env.sh; cd $S
until [ $(ls $B/*/uniform3_s*/loop.json 2>/dev/null | wc -l) -ge 86 ]; do sleep 60; done
python scripts/vlm_clip_score.py paper/1012/vlm_scores_bench43.json "uniform3=lingbot-world-v2/specwm_out/bench43/*/uniform3_s*/video.mp4" > $B/vlm_uniform3.txt 2>&1
echo "$(date) U3_SCORED" >> $B/vlm_uniform3.txt
