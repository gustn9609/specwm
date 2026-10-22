#!/bin/bash
# [10/17 frozen baselines] helper for score_runs.sh: run "$@" while polling the memory used on GPU $1 every 2 s; write "held_s peak_MiB" to $2.
# Called INSIDE the flock (flock /tmp/gpu<G>.lock env CUDA_VISIBLE_DEVICES=<G> ... _gpu_peak_run.sh G PEAKFILE cmd...), so the hold time and the
# peak cover only our own lock hold, not the queue wait (whole-GPU memory from nvidia-smi; exclusive while the lock is respected).
G=$1; OUT=$2; shift 2; T=$(mktemp); nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G" -l 2 > "$T" 2>/dev/null & P=$!
t0=$(date +%s); "$@"; rc=$?
kill $P 2>/dev/null; wait $P 2>/dev/null; echo "$(( $(date +%s) - t0 )) $(sort -n "$T" 2>/dev/null | tail -1)" > "$OUT"; rm -f "$T"; exit $rc
