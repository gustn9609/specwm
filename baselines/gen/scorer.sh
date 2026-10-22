#!/bin/bash
# [10/17 frozen baselines] Scorer loop: every --sleep s (600) find every runs/<model_arm>/ with clips lacking a metric (gen/pending_arms.py) and run
# baselines/score_runs.sh ARM_DIR GPU on it: judge (Qwen3-VL on the GPU-7 server :JUDGE_PORT, under flock /tmp/vlm_base.lock; CPU here), vbench6
# --fps-norm 16 and rpe --program-from-path (both under flock /tmp/gpu<GPU>.lock in batches of 8 clips, hold + peak GPU memory logged by
# _gpu_peak_run.sh), reaction on the *__P5* clips (CPU). Only clips with timing.json whose video.mp4 is older than --min-age s are scored (a clip still
# being written must never reach the append-only metrics files). The loop exits when the STOP file exists AND nothing is pending (after a final pass).
# usage: /home/hk28965/specwm/baselines/gen/scorer.sh GPU [--dry] [--once] [--sleep 600] [--min-age 120] [--stop FILE] [--judge-port 30100] [--batch 8]
#   GPU    = the index assigned by the orchestrator (the lock file /tmp/gpu<GPU>.lock is taken by score_runs.sh for every GPU batch, never here)
#   --dry  = print the score_runs.sh commands that WOULD run (one pass, no GPU, no judge), --once = one real pass then exit
#   STOP   = /home/hk28965/specwm/baselines/gen/STOP_SCORER by default (touch it to let the loop finish and exit)
# log: /home/hk28965/specwm/baselines/gen/logs/scorer.log (+ score_runs.sh's own baselines/logs/score_<arm>.log); summary table: python3 baselines/report.py
set -u
S=/home/hk28965/specwm; B=$S/baselines; GEN=$B/gen
G=${1:?usage: scorer.sh GPU [--dry] [--once] [--sleep S] [--min-age S] [--stop FILE] [--judge-port P] [--batch N]}; shift
DRY=0; ONCE=0; SLEEP=600; MINAGE=120; STOP=$GEN/STOP_SCORER; JP=${JUDGE_PORT:-30100}; BATCH=8
while [ $# -gt 0 ]; do case $1 in
  --dry) DRY=1; shift;; --once) ONCE=1; shift;; --sleep) SLEEP=$2; shift 2;; --min-age) MINAGE=$2; shift 2;; --stop) STOP=$2; shift 2;;
  --judge-port) JP=$2; shift 2;; --batch) BATCH=$2; shift 2;; *) echo "unknown option $1"; exit 1;; esac; done
mkdir -p $GEN/logs; LOG=$GEN/logs/scorer.log
log() { echo "[$(date '+%m/%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "scorer start: gpu $G dry $DRY once $ONCE sleep $SLEEP s min-age $MINAGE s judge-port $JP batch $BATCH stop-file $STOP pid $$"
PASS=0
while :; do
  PASS=$((PASS + 1)); PEND=$(python3 $GEN/pending_arms.py --min-age "$MINAGE" 2>>"$LOG")
  if [ -z "$PEND" ]; then log "pass $PASS: nothing pending"
  else
    log "pass $PASS: pending ->"$'\n'"$PEND"
    while IFS=$'\t' read -r ARM N READY; do
      [ -n "$ARM" ] || continue
      CMD="$B/score_runs.sh $B/runs/$ARM $G --judge-port $JP --fps-norm 16 --batch $BATCH --min-age $MINAGE"
      if [ "$DRY" = 1 ]; then log "DRY: $CMD   # $N of $READY clips pending"
      else
        log "scoring $ARM ($N of $READY clips pending): $CMD"; t0=$(date +%s)
        $CMD >> "$LOG" 2>&1; rc=$?; log "scored $ARM in $(( $(date +%s) - t0 )) s (rc $rc)"
      fi
    done <<< "$PEND"
  fi
  if [ "$DRY" = 1 ] || [ "$ONCE" = 1 ]; then log "scorer exit (dry $DRY once $ONCE)"; exit 0; fi
  if [ -f "$STOP" ]; then
    LEFT=$(python3 $GEN/pending_arms.py --min-age "$MINAGE" 2>>"$LOG")
    if [ -z "$LEFT" ]; then log "STOP file $STOP present and nothing pending -> exit"; exit 0; fi
    log "STOP file present but still pending (will exit when scored):"$'\n'"$LEFT"
  fi
  sleep "$SLEEP"
done
