#!/bin/bash
# head retrained without bench-43 scenes -> teacher-parity mode (cw07_safe) on bench-43 with that head -> VLM score + paired comparison
S=/home/hk28965/specwm; O=$S/lingbot-world-v2/specwm_out; D=$O/gap_dump_nob43; SC=/tmp/claude-1741758895/-home-hk28965/76e7a8ef-ae2e-455b-8c73-052f5bde2a30/scratchpad
cd $S; source env.sh
T43=$(cat $SC/t43.txt); A=$(echo $T43 | cut -d, -f1-22); B=$(echo $T43 | cut -d, -f23-)
i=0; for s in 1234 4321; do for tl in $A $B; do G=$(( (i % 4) )); G=$(echo "0 1 5 6" | cut -d' ' -f$((G+1)))
  CUDA_VISIBLE_DEVICES=$G SAVE_LAT=0 RESERVE_GB=85 nohup bash scripts/envrun.sh python -u -m specwm.spec_early --mode diag:learned_dense --eps 0.24 --no-ref --chunk-frac 0.7 --commit safe --arm cw07_safe_nob43 --head $O/head14b_nob43/head.pt --tasks $tl --seeds $s --out specwm_out/bench43 > $O/bench43/logs/cw07_safe_nob43_${s}_$i.log 2>&1 &
  i=$((i+1)); done; done; wait
python scripts/vlm_clip_score.py paper/1012/vlm_scores_bench43.json "cw07_safe_nob43=lingbot-world-v2/specwm_out/bench43/*/cw07_safe_nob43_s*/video.mp4" > $O/bench43/vlm_nob43.txt 2>&1
echo "$(date) NOB43_DONE"
