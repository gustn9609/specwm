#!/bin/bash
# [10/17] split scoring across two GPUs without concurrent writes to one metrics file:
#   scorer_split.sh GPU METRICS_DIR SKIP ARM_GLOB...   loops over runs/<arm> dirs matching the globs, calls score_runs.sh for each arm with pending clips.
#   e.g. GPU 7 (shares the judge GPU; ~12 GB free): baseline arms -> metrics_g7/ ; GPU 6: ours_* arms -> metrics/
# exits when gen/STOP_SCORER exists and a full pass found nothing to do.
B=/home/hk28965/specwm/baselines; G=$1; M=$2; SKIP=$3; shift 3; LOG=$B/gen/logs/scorer_split_g$G.log; mkdir -p $M
log() { echo "[$(date +%m/%d\ %H:%M:%S)] $*" >> $LOG; }
log "start gpu $G metrics $M skip '$SKIP' arms: $*"
while true; do
  did=0
  for pat in "$@"; do for d in $B/runs/$pat; do [ -d "$d" ] || continue
    before=$(cat $M/*.json 2>/dev/null | wc -c)
    $B/score_runs.sh $d $G --metrics $M --skip "$SKIP" --judge-port 30100 --fps-norm 16 --batch 8 --min-age 120 >> $LOG 2>&1
    after=$(cat $M/*.json 2>/dev/null | wc -c); [ "$after" != "$before" ] && { did=1; log "scored $(basename $d)"; }
  done; done
  if [ $did = 0 ]; then [ -e $B/gen/STOP_SCORER ] && { log "STOP and nothing pending -> exit"; exit 0; }; sleep 600; fi
done
