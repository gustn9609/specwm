#!/bin/bash
# [10/21] gen/worker.sh for a GPU that cannot take PAIR chunks (GPUs outside the pairs 0+1 / 2+3, e.g. 5 and 6): identical to worker.sh except that
# next_chunk() skips the .sh chunks whose header (first 4 lines) has "# HYB_CHUNK ..." (pair chunks of make_queue_*.py); .json chunks and "# OURS_CHUNK" / baseline .sh
# chunks are claimed as usual. Everything else (locks, STOP files, logs, timeouts, done/failed) is worker.sh verbatim.
# usage:  nohup /home/hk28965/specwm/baselines/gen/worker_solo.sh GPU > /dev/null 2>&1 &
G=${1:?usage: worker_solo.sh GPU}; S=/home/hk28965/specwm; B=$S/baselines; GEN=$B/gen; Q=${QUEUE_ROOT:-$GEN/queue}; LOGS=$GEN/logs
POLL=${POLL:-600}; MAXIDLE=${MAXIDLE:-3}; MINFREE_GB=${MINFREE_GB:-8}
mkdir -p "$LOGS" "$Q/claimed" "$Q/done" "$Q/failed"; WLOG=$LOGS/worker$G.log
log() { echo "[$(date +%F_%T)] gpu$G $*" | tee -a "$WLOG"; }
free_gb() { df -BG --output=avail "$B" | tail -1 | tr -dc 0-9; }
# [10/21 18:50] other users' jobs appeared on GPUs 0-3,5,6 -> (1) nothing is claimed while the GPU (and the pair partner, for HYB_CHUNK) has < MINFREE_GPU_MB free
# (default 50000 MiB; the worker logs and waits POLL s, not counted as idle); (2) a chunk whose output shows a CUDA OOM / "cloud died before listening" goes back to its tier
# (queue/retry_count/<name>, up to MAX_RETRY=3 times) instead of failed/.
gpu_free_mb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" 2>/dev/null | awk -F', ' '{print $2-$1}'; }
MINFREE_GPU_MB=${MINFREE_GPU_MB:-50000}; MAX_RETRY=${MAX_RETRY:-3}; case $G in 0) PARTNER=1;; 1) PARTNER=0;; 2) PARTNER=3;; 3) PARTNER=2;; *) PARTNER=;; esac
next_chunk() {   # first unclaimed NON-PAIR chunk: tiers in numeric order, names in C-locale order inside a tier
  local t f
  for t in $(ls "$Q" | grep -E '^[0-9]+$' | sort -n); do
    for f in $(ls "$Q/$t"/*.json "$Q/$t"/*.sh 2>/dev/null | LC_ALL=C sort); do
      case $f in *.sh) head -4 "$f" 2>/dev/null | grep -q '^# HYB_CHUNK ' && continue;; esac
      echo "$f"; return 0
    done
  done; return 1
}
log "worker_solo start (skips HYB_CHUNK pair chunks): queue $Q poll ${POLL}s maxidle $MAXIDLE minfree ${MINFREE_GB}GB pid $$"
idle=0
while :; do
  [ -f "$Q/STOP.gpu$G" ] && { log "STOP.gpu$G present -> exit"; exit 0; }
  f=$(next_chunk)
  if [ -z "$f" ]; then
    [ -f "$Q/STOP" ] && { log "queue empty and STOP present -> exit"; exit 0; }
    idle=$((idle + 1)); log "queue empty (no solo chunk): idle poll $idle/$MAXIDLE"
    [ $idle -ge $MAXIDLE ] && { log "exit after $MAXIDLE idle polls"; exit 0; }
    sleep "$POLL"; continue
  fi
  fr=$(free_gb)
  if [ -z "$fr" ] || [ "$fr" -lt "$MINFREE_GB" ]; then log "free disk ${fr:-?} GB < ${MINFREE_GB} GB: not claiming, waiting ${POLL}s"; sleep "$POLL"; continue; fi
  gf=$(gpu_free_mb $G); pf=$gf; if [ -n "$PARTNER" ]; then case $f in *.sh) head -4 "$f" 2>/dev/null | grep -q '^# HYB_CHUNK ' && pf=$(gpu_free_mb $PARTNER);; esac; fi
  if { [ -n "$gf" ] && [ "$gf" -lt "$MINFREE_GPU_MB" ]; } || { [ -n "$pf" ] && [ "$pf" -lt "$MINFREE_GPU_MB" ]; }; then log "gpu$G free ${gf:-?} MiB (partner ${pf:-?}) < ${MINFREE_GPU_MB}: other users' jobs on the GPU, not claiming $(basename "$f"), waiting ${POLL}s"; sleep "$POLL"; continue; fi
  name=$(basename "$f"); tier=$(basename "$(dirname "$f")"); c=$Q/claimed/$name.gpu$G
  mv -- "$f" "$c" 2>/dev/null || { log "claim race on $name (taken by another worker)"; continue; }
  OUT=$LOGS/$name.gpu$G.out; t0=$(date +%s); held=na; peak=na; rc=99
  case $name in
    *.json)
      read SET ARM N <<< "$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['set'], d['arm'], len(d.get('ids') or d['tasks']))" "$c" 2>/dev/null)"
      if [ -z "$ARM" ]; then log "FAILED $name: not a chunk json with set/arm/tasks"; mv -- "$c" "$Q/failed/$name.gpu$G"; echo "bad chunk json" > "$Q/failed/$name.gpu$G.log"; continue; fi
      log "claimed tier $tier $name: run_ours.sh $SET $ARM gpu $G ($N tasks) -> $OUT"
      PK=$(mktemp); flock /tmp/gpu$G.lock env CUDA_VISIBLE_DEVICES=$G "$B/_gpu_peak_run.sh" "$G" "$PK" "$B/run_ours.sh" "$SET" "$ARM" "$G" "$c" > "$OUT" 2>&1; rc=$?
      read held peak < "$PK" 2>/dev/null; rm -f "$PK"; peak=${peak:-na}; held=${held:-na}
      grep -h "OURS_DONE\|^\[batch\]" "$OUT" 2>/dev/null | tail -2 | sed "s/^/[$(date +%F_%T)] gpu$G   /" | tee -a "$WLOG";;
    *.sh)
      log "claimed tier $tier $name: bash $name $G -> $OUT"
      setsid bash "$c" "$G" > "$OUT" 2>&1 & cpid=$!; tlim=$(( $(date +%s) + ${CHUNK_TIMEOUT:-14400} )); rc=""
      while kill -0 $cpid 2>/dev/null; do [ $(date +%s) -ge $tlim ] && { log "TIMEOUT $name after ${CHUNK_TIMEOUT:-14400}s -> killing process group $cpid"; kill -TERM -- -$cpid 2>/dev/null; sleep 20; kill -KILL -- -$cpid 2>/dev/null; rc=124; break; }; sleep 15; done
      if [ -z "$rc" ]; then wait $cpid; rc=$?; fi;;
    *) log "FAILED $name: unknown chunk type"; mv -- "$c" "$Q/failed/$name.gpu$G"; echo "unknown chunk type" > "$Q/failed/$name.gpu$G.log"; continue;;
  esac
  wall=$(( $(date +%s) - t0 )); hp="lock-hold ${held}s peak ${peak} MiB"; [ "$held" = na ] && hp="(hold/peak logged by the chunk script itself)"
  if [ "$rc" -eq 0 ]; then mv -- "$c" "$Q/done/$name.gpu$G"; log "DONE $name rc 0 wall ${wall}s $hp"
  elif grep -qE "CUDA error: out of memory|CUDA out of memory|OutOfMemoryError|cloud died before listening" "$OUT" 2>/dev/null && { mkdir -p "$Q/retry_count"; nr=$(( $(cat "$Q/retry_count/$name" 2>/dev/null || echo 0) + 1 )); echo $nr > "$Q/retry_count/$name"; [ $nr -le $MAX_RETRY ]; }; then
    mv -- "$c" "$Q/$tier/$name"; log "OOM on $name rc $rc wall ${wall}s (retry $nr/$MAX_RETRY) -> back to queue/$tier, waiting ${POLL}s"; sleep "$POLL"; continue
  else mv -- "$c" "$Q/failed/$name.gpu$G"; tail -n 60 "$OUT" > "$Q/failed/$name.gpu$G.log" 2>/dev/null; log "FAILED $name rc $rc wall ${wall}s $hp -> failed/$name.gpu$G.log"; fi
  idle=0
done
