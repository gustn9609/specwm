#!/bin/bash
# [10/17 B frozen baselines] Generation worker for ONE GPU: claims chunks from the queue in tier order and runs them under that GPU's lock.
# usage:  nohup /home/hk28965/specwm/baselines/gen/worker.sh GPU > /dev/null 2>&1 &      (one worker per GPU; several workers on different GPUs coexist:
#         the claim is an atomic rename, so a chunk is run by exactly one worker)
# queue layout (QUEUE_ROOT, default baselines/gen/queue):
#   <tier>/<NNN>_<set>_<arm>.json   our chunks (baselines/gen/make_queue.py: {"set","arm","tasks":[...]}): run as
#                                   flock /tmp/gpu<G>.lock env CUDA_VISIBLE_DEVICES=<G> baselines/_gpu_peak_run.sh G PEAK baselines/run_ours.sh SET ARM G CHUNK
#                                   (run_ours.sh takes NO lock itself; _gpu_peak_run.sh adds the lock-hold seconds and the nvidia-smi peak MiB of the hold)
#   <tier>/*.sh                     baseline job files written by the baseline agents: run as  bash FILE G  (the .sh takes its own flock on /tmp/gpu$G.lock)
#   claimed/<name>.gpu<G>  done/<name>.gpu<G>  failed/<name>.gpu<G> (+ failed/<name>.gpu<G>.log = last 60 lines of the run output)
#   STOP (in the queue root): exit once the queue is empty;  STOP.gpu<G>: exit before claiming the next chunk (the running chunk finishes)
# order: tier dirs numerically (1, 2, 3, ...), inside a tier by file name (NNN prefix); .json and .sh chunks of a tier are interleaved by name.
# idle: when nothing is claimable the worker polls every POLL s (default 600) and exits after MAXIDLE (default 3) idle polls, or at once if STOP exists.
# disk: when the free space of /home is below MINFREE_GB (default 8) nothing is claimed; the worker logs and waits POLL s (does not count as idle).
# logs: baselines/gen/logs/worker<G>.log (this worker), baselines/gen/logs/<name>.gpu<G>.out (full output of each chunk run)
# never kills anything; never touches other GPUs; hold time > 20 min is logged as WARN (chunk sizes are the orchestrator's; see make_queue.py)
G=${1:?usage: worker.sh GPU}; S=/home/hk28965/specwm; B=$S/baselines; GEN=$B/gen; Q=${QUEUE_ROOT:-$GEN/queue}; LOGS=$GEN/logs
POLL=${POLL:-600}; MAXIDLE=${MAXIDLE:-3}; MINFREE_GB=${MINFREE_GB:-8}
mkdir -p "$LOGS" "$Q/claimed" "$Q/done" "$Q/failed"; WLOG=$LOGS/worker$G.log
log() { echo "[$(date +%F_%T)] gpu$G $*" | tee -a "$WLOG"; }
free_gb() { df -BG --output=avail "$B" | tail -1 | tr -dc 0-9; }
# [10/21 18:50] other users' jobs appeared on GPUs 0-3,5,6 -> (1) nothing is claimed while the GPU (and the pair partner, for HYB_CHUNK) has < MINFREE_GPU_MB free
# (default 50000 MiB; the worker logs and waits POLL s, not counted as idle); (2) a chunk whose output shows a CUDA OOM / "cloud died before listening" goes back to its tier
# (queue/retry_count/<name>, up to MAX_RETRY=3 times) instead of failed/.
gpu_free_mb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" 2>/dev/null | awk -F', ' '{print $2-$1}'; }
MINFREE_GPU_MB=${MINFREE_GPU_MB:-50000}; MAX_RETRY=${MAX_RETRY:-3}; case $G in 0) PARTNER=1;; 1) PARTNER=0;; 2) PARTNER=3;; 3) PARTNER=2;; *) PARTNER=;; esac
next_chunk() {   # first unclaimed chunk: tiers in numeric order, names in C-locale order inside a tier
  local t f
  for t in $(ls "$Q" | grep -E '^[0-9]+$' | sort -n); do
    f=$(ls "$Q/$t"/*.json "$Q/$t"/*.sh 2>/dev/null | LC_ALL=C sort | head -1); [ -n "$f" ] && { echo "$f"; return 0; }
  done; return 1
}
log "worker start: queue $Q poll ${POLL}s maxidle $MAXIDLE minfree ${MINFREE_GB}GB pid $$"
idle=0
while :; do
  [ -f "$Q/STOP.gpu$G" ] && { log "STOP.gpu$G present -> exit"; exit 0; }
  f=$(next_chunk)
  if [ -z "$f" ]; then
    [ -f "$Q/STOP" ] && { log "queue empty and STOP present -> exit"; exit 0; }
    idle=$((idle + 1)); log "queue empty: idle poll $idle/$MAXIDLE"
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
      # [10/19] .sh chunks run in their own process group with a wall-clock limit (CHUNK_TIMEOUT s, default 5400): two hybrid
      # neutral_abot/neutral_832 chunks hung for hours after their last clip (10/18); on timeout the whole group is killed -> rc 124 -> failed/
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
  [ "$held" != na ] && [ "$held" -gt 1200 ] 2>/dev/null && log "WARN lock hold ${held}s > 20 min on $name (reduce the chunk size in make_queue.py SIZES)"
  idle=0
done
