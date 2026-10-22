#!/bin/bash
# [10/21] Scene-persistence scorer over run dirs (scripts/scene_persist.py: CLIP/DINOv2 similarity to the first chunk + zero-shot content class per second).
# usage: gen/scene_persist_runs.sh GPU OUT_JSON RUN_DIR_GLOB...      e.g.  gen/scene_persist_runs.sh 6 metrics_sp/scene_persist.json 'runs/*_long'
# Collects every <run dir>/<clip>/video.mp4 that has a timing.json next to it and is older than 120 s (not still being written), runs the scorer under
# flock /tmp/gpu<GPU>.lock (skip-if-scored: the json is append-only, keyed by the absolute video path). Log: gen/logs/scene_persist_g<GPU>.log
G=${1:?GPU}; OUT=${2:?OUT_JSON}; shift 2; B=/home/hk28965/specwm/baselines; S=/home/hk28965/specwm; LOG=$B/gen/logs/scene_persist_g$G.log; mkdir -p "$(dirname "$OUT")" $B/gen/logs
cd $B; V=(); now=$(date +%s)
for pat in "$@"; do for d in $pat; do [ -d "$d" ] || continue
  for v in "$d"/*/video.mp4; do [ -f "$v" ] && [ -f "$(dirname "$v")/timing.json" ] && [ $(( now - $(stat -c %Y "$v") )) -ge 120 ] && V+=("$v"); done
done; done
echo "[$(date +%F_%T)] scene_persist gpu $G: ${#V[@]} videos from $* -> $OUT" | tee -a "$LOG"
[ ${#V[@]} -gt 0 ] || exit 0
flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G $S/tools/vbench_env/bin/python $S/scripts/scene_persist.py "$OUT" "${V[@]}" 2>&1 | grep -E "^\[scene_persist\]" >> "$LOG"; rc=${PIPESTATUS[0]}
echo "[$(date +%F_%T)] scene_persist gpu $G: rc $rc" | tee -a "$LOG"; exit $rc
